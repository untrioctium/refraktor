#include <librefrakt/image/denoiser.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util.h>

#include <spdlog/spdlog.h>

#define OPTIX_DONT_INCLUDE_CUDA
using CUcontext = RUcontext;
using CUstream = RUstream;

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <optix_denoiser_tiling.h>

#define CHECK_OPTIX(expr) \
	do { \
		if (auto result = expr; result != OPTIX_SUCCESS) { \
			SPDLOG_CRITICAL("'{}' failed with '{}'", #expr, optixGetErrorName(result)); exit(1); \
		} \
	} while (0)


class rfkt::denoiser_old::denoiser_impl {
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
		std::swap(tiling, d.tiling);
		std::swap(tile_size, d.tile_size);

		return *this;
	}

	static std::unique_ptr<denoiser_impl> create(uint2 max_dims, denoiser_flag::flags options) {

		auto d = denoiser_impl{};
		d.upscale_2x = options & denoiser_flag::upscale;
		d.tiling = options & denoiser_flag::tiled;

		if (d.tiling) {
			d.tile_size = max_dims;
		}

		else if (d.upscale_2x) {
			max_dims.x /= 2;
			max_dims.y /= 2;
		}

		OptixDenoiserOptions denoiser_options = {};
		denoiser_options.guideAlbedo = 0;
		denoiser_options.guideNormal = 0;
		denoiser_options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

		CHECK_OPTIX(optixDenoiserCreate(optix_context, (d.upscale_2x)? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_LDR, &denoiser_options, &d.handle));

		memset(&d.szs, 0, sizeof(OptixDenoiserSizes));
		CHECK_OPTIX(optixDenoiserComputeMemoryResources(d.handle, max_dims.x, max_dims.y, &d.szs));

		if (d.tiling) {
			max_dims.x += 2 * d.szs.overlapWindowSizeInPixels;
			max_dims.y += 2 * d.szs.overlapWindowSizeInPixels;
		}

		d.state_buffer = roccu::gpu_buffer<>(d.szs.stateSizeInBytes);
		auto scratch_size = d.tiling ? d.szs.withOverlapScratchSizeInBytes : d.szs.withoutOverlapScratchSizeInBytes;
		d.scratch_buffer = roccu::gpu_buffer<>(scratch_size);

		SPDLOG_INFO("Denoiser state sizes: {}mb, {}mb", d.state_buffer.size_bytes() / (1024 * 1024), d.scratch_buffer.size_bytes() / (1024 * 1024));

		CHECK_OPTIX(optixDenoiserSetup(
			d.handle, 0, 
			max_dims.x, max_dims.y, 
			d.state_buffer.ptr(), d.state_buffer.size_bytes(), 
			d.scratch_buffer.ptr(), d.scratch_buffer.size_bytes()));

		d.dp.blendFactor = 0;
		d.dp.hdrIntensity = 0;
		//cuMemAlloc(&d.dp.hdrIntensity, sizeof(float));

		return std::unique_ptr<denoiser_impl>(new denoiser_impl{std::move(d)});
	}

	std::future<double> denoise(const image_type& in, image_type& out, roccu::gpu_stream& stream) {

		memset(&layer, 0, sizeof(layer));

		layer.input.width = in.dims().x;
		layer.input.height = in.dims().y;
		layer.input.rowStrideInBytes = in.width() * sizeof(pixel_type);
		layer.input.pixelStrideInBytes = sizeof(pixel_type);
		//layer.input.format = OPTIX_PIXEL_FORMAT_HALF3;

		if constexpr(std::is_same_v<pixel_type, float3>) {
			layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT3;
		} else if constexpr(std::is_same_v<pixel_type, float4>) {
			layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
		} else if constexpr(std::is_same_v<pixel_type, uint8_t>) {
			layer.input.format = OPTIX_PIXEL_FORMAT_UCHAR3;
		}
		else if constexpr (std::is_same_v<pixel_type, half3>) {
			layer.input.format = OPTIX_PIXEL_FORMAT_HALF3;
		}

		layer.output = layer.input;

		layer.output.width = out.dims().x;
		layer.output.height = out.dims().y;
		layer.output.rowStrideInBytes = out.width() * sizeof(pixel_type);

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

		if (!tiling) {
			CHECK_OPTIX(
				optixDenoiserInvoke(
					handle, stream, &dp,
					state_buffer.ptr(), state_buffer.size_bytes(),
					&guide_layer, &layer, 1, 0, 0,
					scratch_buffer.ptr(), scratch_buffer.size_bytes()
				)
			);
		}
		else {
			CHECK_OPTIX(
				optixUtilDenoiserInvokeTiled(
					handle, stream, &dp,
					state_buffer.ptr(), state_buffer.size_bytes(),
					&guide_layer, &layer, 1,
					scratch_buffer.ptr(), scratch_buffer.size_bytes(),
					szs.overlapWindowSizeInPixels, tile_size.x, tile_size.y
				)
			);
		}
		stream.host_func(
			[timer=std::move(timer), promise = std::move(promise)]() mutable {
			promise.set_value(timer->count());
		});

		return future;
	}

	~denoiser_impl() {
		if (handle != nullptr) {
			optixDenoiserDestroy(handle);
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

	roccu::gpu_buffer<> state_buffer;
	roccu::gpu_buffer<> scratch_buffer;

	bool upscale_2x;
	bool tiling;

	uint2 tile_size;
};

rfkt::denoiser_old::~denoiser_old() = default;
rfkt::denoiser_old::denoiser_old(uint2 max_dims, denoiser_flag::flags options) : impl(denoiser_impl::create(max_dims, options)) {}
rfkt::denoiser_old& rfkt::denoiser_old::operator=(denoiser_old&& d) noexcept {
	std::swap(impl, d.impl);
	return *this;
}

rfkt::denoiser_old::denoiser_old(denoiser_old&& d) noexcept {
	(*this) = std::move(d);
}


std::future<double> rfkt::denoiser_old::denoise(roccu::gpu_image_view<half3> in, roccu::gpu_image_view<half3> out, roccu::gpu_stream& stream) {
	return impl->denoise(in, out, stream);
}

void rfkt::denoiser_old::init(CUcontext ctx) {
	rfkt::denoiser_old::denoiser_impl::init(ctx);
}

unsigned short rand_uniform_half() {
	constexpr static unsigned short min_val = 0x0400;
	constexpr static unsigned short max_val = 0x3c00;
	constexpr static auto range = max_val - min_val;

	return rand() % range + min_val;
}

double rfkt::denoiser_old::benchmark(uint2 dims, denoiser_flag::flags options, std::uint32_t num_passes, roccu::gpu_stream& stream)
{
	auto input_dims = dims;
	auto input_size = dims.x * dims.y;

	if (options & denoiser_flag::upscale) {
		input_size /= 4;
		input_dims.x /= 2;
		input_dims.y /= 2;
	}

	auto dn = rfkt::denoiser_old{ dims, options };

	auto input = roccu::gpu_image<half3>{ input_dims };
	auto output = roccu::gpu_image<half3>{ dims };

	auto input_size_nshorts = input_size * 3;

	auto bytes = std::vector<unsigned short>(input_size_nshorts);
	for (auto& b : bytes) {
		b = rand_uniform_half();
	}

	ruMemcpyHtoD(input.ptr(), bytes.data(), sizeof(unsigned short) * bytes.size());

	double sum = 0.0;

	for (int i = 0; i < num_passes; i++) {
		sum += dn.denoise(input, output, stream).get();
	}

	return sum / num_passes;

}