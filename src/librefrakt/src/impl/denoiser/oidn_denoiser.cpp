#include <spdlog/spdlog.h>

#include <OpenImageDenoise/oidn.hpp>

#include <librefrakt/interface/denoiser.h>

namespace rfkt {

	struct oidn_denoiser : public denoiser::registrar<oidn_denoiser> {
		const static inline meta_type meta = {
			.pretty_name = "Intel OpenImageDenoise",
			.priority = 10,
			.supported_apis = { ROCCU_API_CUDA, ROCCU_API_ROCM },
			.upscale_supported = false
		};

		oidn_denoiser(uint2 dims, denoiser_flag::flags options, roccu::gpu_stream& stream): stream(stream) {
			device = [&stream]() -> oidn::DeviceRef {
				if (roccuGetApi() == ROCCU_API_CUDA) {
					return oidn::newCUDADevice({ 0 }, { (cudaStream_t)stream.operator RUstream_st * () });
				}
				else {
					return oidn::newHIPDevice({ 0 }, { (hipStream_t)stream.operator RUstream_st * () });
				}
			}();

			device.commit();
			check_error();
			filter = device.newFilter("RT");

			check_error();
		}

		std::future<double> denoise(const image_type& in, image_type& out, roccu::gpu_event& event) override {

			oidn::BufferRef in_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(in.ptr()), in.size_bytes());
			oidn::BufferRef out_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(out.ptr()), out.size_bytes());

			filter.setImage("color", in_buf, oidn::Format::Half3, in.dims().x, in.dims().y);
			filter.setImage("output", out_buf, oidn::Format::Half3, out.dims().x, out.dims().y);
			oidnSetFilterInt(filter.getHandle(), "quality", OIDN_QUALITY_BALANCED);
			filter.commit();
			check_error();

			auto timer = std::make_shared<rfkt::timer>();
			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func([timer]() { timer->reset(); });

			filter.execute();
			stream.record(event);

			stream.host_func([timer = std::move(timer), promise = std::move(promise)]() mutable {
				promise.set_value(timer->count());
			});

			return future;
		}

		void check_error() {
			if (const char* error; device.getError(error) != oidn::Error::None) {
				SPDLOG_ERROR("OIDN error: {}", error);
				__debugbreak();
			}
		}

		roccu::gpu_stream& stream;
		oidn::DeviceRef device;
		oidn::FilterRef filter;
	};

}