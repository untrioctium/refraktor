#include <librefrakt/image/denoiser.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util.h>

#include <spdlog/spdlog.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define CHECK_OPTIX(expr) if(auto result = expr; result != OPTIX_SUCCESS) { SPDLOG_CRITICAL("'{}' failed with '{}'", #expr, optixGetErrorName(result)); exit(1); }


class rfkt::denoiser::denoiser_impl {
public:

	static void init(CUcontext ctx) {
		CHECK_OPTIX(optixInit());
		CHECK_OPTIX(optixDeviceContextCreate(ctx, nullptr, &optix_context));
	}

	denoiser_impl(const denoiser_impl&) = delete;
	denoiser_impl& operator=(const denoiser_impl&) = delete;

	denoiser_impl(denoiser_impl&& d) noexcept {
		(*this) = std::move(d);
	}

	denoiser_impl& operator=(denoiser_impl&& d) noexcept {
		std::swap(handle, d.handle);
		std::swap(state_buffer, d.state_buffer);
		std::swap(scratch_buffer, d.scratch_buffer);
		std::swap(szs, d.szs);
		std::swap(dp, d.dp);
		std::swap(layer, d.layer);
		std::swap(guide_layer, d.guide_layer);
		std::swap(upscale_2x, d.upscale_2x);

		return *this;
	}

	static std::unique_ptr<denoiser_impl> create(uint2 max_dims, bool upscale_2x) {

		auto d = denoiser_impl{};
		d.upscale_2x = upscale_2x;

		if (upscale_2x) {
			max_dims.x /= 2;
			max_dims.y /= 2;
		}

		OptixDenoiserOptions denoiser_options = {};
		denoiser_options.guideAlbedo = 0;
		denoiser_options.guideNormal = 0;

		CHECK_OPTIX(optixDenoiserCreate(optix_context, (d.upscale_2x)? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_LDR, &denoiser_options, &d.handle));

		memset(&d.szs, 0, sizeof(OptixDenoiserSizes));
		CHECK_OPTIX(optixDenoiserComputeMemoryResources(d.handle, max_dims.x, max_dims.y, &d.szs));

		CUDA_SAFE_CALL(cuMemAlloc(&d.state_buffer, d.szs.stateSizeInBytes));
		CUDA_SAFE_CALL(cuMemAlloc(&d.scratch_buffer, d.szs.withoutOverlapScratchSizeInBytes));

		CHECK_OPTIX(optixDenoiserSetup(d.handle, 0, max_dims.x, max_dims.y, d.state_buffer, d.szs.stateSizeInBytes, d.scratch_buffer, d.szs.withoutOverlapScratchSizeInBytes));

		d.dp.blendFactor = 0;
		d.dp.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
		d.dp.hdrIntensity = 0;
		//cuMemAlloc(&d.dp.hdrIntensity, sizeof(float));

		return std::unique_ptr<denoiser_impl>(new denoiser_impl{std::move(d)});
	}

	std::future<double> denoise(uint2 dims, cuda_span<half3> in, cuda_span<half3> out, cuda_stream& stream) {

		memset(&layer, 0, sizeof(layer));

		layer.input.width = dims.x;
		layer.input.height = dims.y;
		layer.input.rowStrideInBytes = dims.x * sizeof(half3);
		layer.input.pixelStrideInBytes = sizeof(half3);
		layer.input.format = OPTIX_PIXEL_FORMAT_HALF3;

		layer.output = layer.input;

		if (upscale_2x) {
			layer.input.width /= 2;
			layer.input.height /= 2;
			layer.input.rowStrideInBytes /= 2;
		}

		layer.input.data = in.ptr();
		layer.output.data = out.ptr();

		auto timer = std::make_shared<rfkt::timer>();
		auto promise = std::promise<double>{};
		auto future = promise.get_future();

		stream.host_func(
			[timer]() {
				timer->reset();
			});

		//CHECK_OPTIX(optixDenoiserComputeIntensity(handle, stream, &layer.input, dp.hdrIntensity, scratch_buffer, szs.withoutOverlapScratchSizeInBytes));
		CHECK_OPTIX(optixDenoiserInvoke(handle, stream, &dp, state_buffer, szs.stateSizeInBytes, &guide_layer, &layer, 1, 0, 0, scratch_buffer, szs.withoutOverlapScratchSizeInBytes));

		stream.host_func(
			[timer=std::move(timer), promise = std::move(promise)]() mutable {
			promise.set_value(timer->count());
		});

		return future;
	}

	~denoiser_impl() {
		if (handle != nullptr) {
			optixDenoiserDestroy(handle);
			cuMemFree(state_buffer);
			cuMemFree(scratch_buffer);
		}
	}

	denoiser_impl() = default;

private:

	inline static OptixDeviceContext optix_context = nullptr;

	OptixDenoiser handle = nullptr;
	OptixDenoiserSizes szs;
	OptixDenoiserParams dp;
	OptixDenoiserLayer layer;
	OptixDenoiserGuideLayer guide_layer = {};

	CUdeviceptr state_buffer = 0;
	CUdeviceptr scratch_buffer = 0;

	bool upscale_2x;
};

rfkt::denoiser::~denoiser() = default;
rfkt::denoiser::denoiser(uint2 max_dims, bool upscale_2x) : impl(denoiser_impl::create(max_dims, upscale_2x)) {}
rfkt::denoiser& rfkt::denoiser::operator=(denoiser&& d) noexcept {
	std::swap(impl, d.impl);
	return *this;
}

rfkt::denoiser::denoiser(denoiser&& d) noexcept {
	(*this) = std::move(d);
}


std::future<double> rfkt::denoiser::denoise(uint2 dims, cuda_span<half3> in, cuda_span<half3> out, cuda_stream& stream) {
	return impl->denoise(dims, in, out, stream);
}

void rfkt::denoiser::init(CUcontext ctx) {
	rfkt::denoiser::denoiser_impl::init(ctx);
}

unsigned short rand_uniform_half() {
	constexpr static unsigned short min_val = 0x0400;
	constexpr static unsigned short max_val = 0x3c00;
	constexpr static auto range = max_val - min_val;

	return rand() % range + min_val;
}

double rfkt::denoiser::benchmark(uint2 dims, bool upscale, std::uint32_t num_passes, cuda_stream& stream)
{
	auto input_dims = dims;
	auto input_size = dims.x * dims.y;
	if (upscale) {
		input_size /= 4;
		input_dims.x /= 2;
		input_dims.y /= 2;
	}

	auto dn = rfkt::denoiser{ dims, upscale };

	auto input = cuda_buffer<half3>{ input_size };
	auto output = cuda_buffer<half3>{ dims.x * dims.y };

	auto input_size_nshorts = input_size * 3;

	auto bytes = std::vector<unsigned short>(input_size_nshorts);
	for (auto& b : bytes) {
		b = rand_uniform_half();
	}

	cuMemcpyHtoD(input.ptr(), bytes.data(), sizeof(unsigned short) * bytes.size());

	double sum = 0.0;

	for (int i = 0; i < num_passes; i++) {
		sum += dn.denoise(input_dims, input, output, stream).get();
	}

	return sum / num_passes;

}