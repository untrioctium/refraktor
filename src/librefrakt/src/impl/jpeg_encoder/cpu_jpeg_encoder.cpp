#include <stb_image_write.h>

#include <librefrakt/interface/jpeg_encoder.h>

namespace rfkt {

	struct cpu_jpeg_encoder : public jpeg_encoder::registrar<cpu_jpeg_encoder> {
		const static inline meta_type meta{
			.priority = 1000,
			.supported_apis = { ROCCU_API_CUDA, ROCCU_API_ROCM}
		};

		cpu_jpeg_encoder(roccu::gpu_stream& stream) {}

		auto encode_image(const gpu_image<uchar3>& image, int quality, roccu::gpu_stream& stream) -> std::future<encode_thunk> override {

			auto local_data = std::make_unique<std::vector<uchar3>>();
			local_data->resize(image.area());

			auto promise = std::promise<encode_thunk>();
			auto future = promise.get_future();

			image.to_host(*local_data, stream);

			stream.host_func([ld = std::move(local_data), promise = std::move(promise), quality, dims = image.dims()]() mutable {
				auto thunk = [ld = std::move(ld), quality, dims]() {
					std::vector<std::byte> ret;

					stbi_write_jpg_to_func(
						[](void* context, void* data, int size) {
							auto& ret = *static_cast<std::vector<std::byte>*>(context);
							ret.insert(ret.end(), static_cast<std::byte*>(data), static_cast<std::byte*>(data) + size);
						}, 
						&ret, dims.x, dims.y, 3, ld->data(), quality);

					return ret;
				};

				promise.set_value(std::move(thunk));
			});

			return future;
		}
	};

}