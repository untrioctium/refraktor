#define NOMINMAX

#include <spdlog/spdlog.h>

#include <librefrakt/interface/denoiser.h>

#define OPTIX_DONT_INCLUDE_CUDA
using CUcontext = RUcontext;
using CUstream = RUstream;

#include <optix.h>
#include <optix_stubs.h>
//#include <optix_function_table_definition.h>
#include <optix_denoiser_tiling.h>

#define CHECK_OPTIX(expr) \
	do { \
		if (auto result = expr; result != OPTIX_SUCCESS) { \
			SPDLOG_CRITICAL("'{}' failed with '{}'", #expr, optixGetErrorName(result)); exit(1); \
		} \
	} while (0)

namespace rfkt {

	struct optix_denoiser : public denoiser::registrar<optix_denoiser> {
		const inline static meta_type meta{
			.pretty_name = "Nvidia OptiX Denoiser",
			.priority = 0,
			.supported_apis = { ROCCU_API_CUDA },
			.upscale_supported = true
		};

		static void init_if_needed() {
			if (optix_context) return;

			CHECK_OPTIX(optixInit());
			CHECK_OPTIX(optixDeviceContextCreate(roccu::context::current(), nullptr, &optix_context));
		}

		optix_denoiser(uint2 dims, denoiser_flag::flags options, roccu::gpu_stream& stream) :
			stream(stream),
			upscale(options & denoiser_flag::upscale),
			tiling(options & denoiser_flag::tiled) {

			init_if_needed();

			if (tiling) {
				tile_size = dims;
			}

			else if (upscale) {
				dims.x /= 2;
				dims.y /= 2;
			}

			auto optix_model = upscale ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X : OPTIX_DENOISER_MODEL_KIND_LDR;
			auto denoiser_options = OptixDenoiserOptions{
				.guideAlbedo = 0,
				.guideNormal = 0,
				.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_DENOISE
			};

			CHECK_OPTIX(optixDenoiserCreate(optix_context, optix_model, &denoiser_options, &handle));

			memset(&szs, 0, sizeof(szs));
			CHECK_OPTIX(optixDenoiserComputeMemoryResources(handle, dims.x, dims.y, &szs));

			if (tiling) {
				dims.x += 2 * szs.overlapWindowSizeInPixels;
				dims.y += 2 * szs.overlapWindowSizeInPixels;
			}

			state_buffer = roccu::gpu_buffer<>{szs.stateSizeInBytes};
			auto scratch_size = tiling ? szs.withOverlapScratchSizeInBytes : szs.withoutOverlapScratchSizeInBytes;
			scratch_buffer = roccu::gpu_buffer<>{scratch_size};

			CHECK_OPTIX(optixDenoiserSetup(
				handle, 0,
				dims.x, dims.y,
				state_buffer.ptr(), state_buffer.size_bytes(),
				scratch_buffer.ptr(), scratch_buffer.size_bytes()));

			memset(&dp, 0, sizeof(dp));
			dp.blendFactor = 0;
			dp.hdrIntensity = 0;
		}

		template<typename PixelType>
		std::future<double> denoise_impl(image_type<PixelType> in, image_type<PixelType> out, roccu::gpu_event& event) {
			memset(&layer, 0, sizeof(layer));

			layer.input.width = in.dims().x;
			layer.input.height = in.dims().y;
			layer.input.rowStrideInBytes = in.pitch() * sizeof(PixelType);
			layer.input.pixelStrideInBytes = sizeof(PixelType);
			
			if constexpr (std::is_same_v<PixelType, half3>) {
				layer.input.format = OPTIX_PIXEL_FORMAT_HALF3;
			}
			else if constexpr (std::is_same_v<PixelType, half4>) {
				layer.input.format = OPTIX_PIXEL_FORMAT_HALF4;
			}
			else if constexpr (std::is_same_v<PixelType, float3>) {
				layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT3;
			}
			else if constexpr (std::is_same_v<PixelType, float4>) {
				layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
			}
			else {
				static_assert(false, "Unsupported pixel type");
			}

			layer.output = layer.input;

			layer.output.width = out.dims().x;
			layer.output.height = out.dims().y;
			layer.output.rowStrideInBytes = out.pitch() * sizeof(PixelType);

			layer.input.data = in.ptr();
			layer.output.data = out.ptr();

			auto timer = std::make_shared<rfkt::timer>();
			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func([timer]() {
				timer->reset();
			});

			if (tiling) {
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
			else {
				CHECK_OPTIX(
					optixDenoiserInvoke(
						handle, stream, &dp,
						state_buffer.ptr(), state_buffer.size_bytes(),
						&guide_layer, &layer, 1, 0, 0,
						scratch_buffer.ptr(), scratch_buffer.size_bytes()
					)
				);
			}

			stream.record(event);

			stream.host_func([timer = std::move(timer), promise = std::move(promise)]() mutable {
				promise.set_value(timer->count());
			});

			return future;
		}

		std::future<double> denoise(image_type<half3> in, image_type<half3> out, roccu::gpu_event& event) override {
			return denoise_impl(in, out, event);
		}

		std::future<double> denoise(image_type<half4> in, image_type<half4> out, roccu::gpu_event& event) override {
			return denoise_impl(in, out, event);
		}

		std::future<double> denoise(image_type<float3> in, image_type<float3> out, roccu::gpu_event& event) override {
			return denoise_impl(in, out, event);
		}

		std::future<double> denoise(image_type<float4> in, image_type<float4> out, roccu::gpu_event& event) override {
			return denoise_impl(in, out, event);
		}

		roccu::gpu_stream& stream;

		bool upscale;
		bool tiling;
		uint2 tile_size;

		inline static OptixDeviceContext optix_context = nullptr;

		OptixDenoiser handle = nullptr;
		OptixDenoiserSizes szs;
		OptixDenoiserParams dp;
		OptixDenoiserLayer layer;
		OptixDenoiserGuideLayer guide_layer = {};

		roccu::gpu_buffer<> state_buffer;
		roccu::gpu_buffer<> scratch_buffer;
	};

}