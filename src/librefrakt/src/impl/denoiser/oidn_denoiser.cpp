#include <spdlog/spdlog.h>

#include <OpenImageDenoise/oidn.hpp>

#include <librefrakt/interface/denoiser.h>
#include <librefrakt/util/filesystem.h>

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

			auto weights_list = rfkt::fs::list(rfkt::fs::working_directory() / "assets/denoise_weights/", rfkt::fs::filter::has_extension(".tza"));
			std::ranges::sort(weights_list, std::less{});

			SPDLOG_INFO("Using OIDN weights: {}", weights_list.back().string());
			weights = rfkt::fs::read_bytes(weights_list.back());

			check_error();
		}

		std::future<double> denoise(const image_type& in, image_type& out, roccu::gpu_event& event) override {

			oidn::BufferRef in_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(in.ptr()), in.size_bytes());
			oidn::BufferRef out_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(out.ptr()), out.size_bytes());

			filter.setImage("color", in_buf, oidn::Format::Half3, in.dims().x, in.dims().y);
			filter.setImage("output", out_buf, oidn::Format::Half3, out.dims().x, out.dims().y);
			filter.setData("weights", weights.data(), weights.size());
			oidnSetFilterInt(filter.getHandle(), "quality", OIDN_QUALITY_BALANCED);
			filter.commit();
			check_error();

			auto timer = std::make_shared<rfkt::timer>();
			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func([timer]() { timer->reset(); });

			filter.execute();
			check_error();
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

		std::vector<char> weights;
		roccu::gpu_stream& stream;
		oidn::DeviceRef device;
		oidn::FilterRef filter;
	};

}