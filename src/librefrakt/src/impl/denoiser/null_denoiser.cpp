#include <librefrakt/interface/denoiser.h>

namespace rfkt {

	struct null_denoiser : public denoiser::registrar<null_denoiser> {
		const static inline meta_type meta{
			.pretty_name = "Null denoiser",
			.priority = std::numeric_limits<decltype(meta_type::priority)>::max(),
			.supported_apis = {ROCCU_API_CUDA, ROCCU_API_ROCM},
			.upscale_supported = false
		};

		null_denoiser(uint2 dims, denoiser_flag::flags options, roccu::gpu_stream& stream): stream(stream) {}

		template<typename PixelType>
		std::future<double> denoise_impl(image_type<PixelType> in, image_type<PixelType> out, roccu::gpu_event& event) {

			auto timer = std::make_shared<rfkt::timer>();
			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func([timer]() { timer->reset(); });
			ruMemcpyDtoDAsync(out.ptr(), in.ptr(), in.size_bytes(), stream);
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
	};

}