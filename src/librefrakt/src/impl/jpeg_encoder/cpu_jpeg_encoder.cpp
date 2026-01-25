
#include <librefrakt/util/stb.hpp>
#include <librefrakt/interface/jpeg_encoder.hpp>

namespace rfkt {

	struct cpu_jpeg_encoder : public jpeg_encoder::registrar<cpu_jpeg_encoder> {
		const static inline meta_type meta{
			.priority = 1000,
			.supported_apis = { ROCCU_API_CUDA, ROCCU_API_ROCM}
		};

		cpu_jpeg_encoder(roccu::gpu_stream& stream) {}

		auto encode_image(roccu::gpu_image_view<uchar3> image, int quality, roccu::gpu_stream& stream) -> std::future<encode_thunk> override {

			auto local_data = std::make_unique<std::vector<uchar3>>();
			local_data->resize(image.area());

			auto promise = std::promise<encode_thunk>();
			auto future = promise.get_future();

			image.to_host(*local_data, stream);

			stream.host_func([ld = std::move(local_data), promise = std::move(promise), quality, width = image.width(), height = image.height()]() mutable {
				auto thunk = [ld = std::move(ld), quality, width, height]() {
					return rfkt::stbi::write_memory(ld->data(), width, height, rfkt::stbi::format::jpg);
				};

				promise.set_value(std::move(thunk));
			});

			return future;
		}
	};

}