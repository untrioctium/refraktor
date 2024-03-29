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

		std::future<double> denoise(const image_type& in, image_type& out, roccu::gpu_event& event) {

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

		roccu::gpu_stream& stream;
	};

}