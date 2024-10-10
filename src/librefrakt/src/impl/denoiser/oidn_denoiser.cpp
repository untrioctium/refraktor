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
			alpha_filter = device.newFilter("RT");

			auto weights_list = rfkt::fs::list(rfkt::fs::working_directory() / "assets/denoise_weights/", rfkt::fs::filter::has_extension(".tza"));
			std::ranges::sort(weights_list, std::less{});

			SPDLOG_INFO("Using OIDN weights: {}", weights_list.back().string());
			weights = rfkt::fs::read_bytes(weights_list.back());

			//filter.setData("weights", weights.data(), weights.size());
			//alpha_filter.setData("weights", weights.data(), weights.size());

			oidnSetFilterInt(filter.getHandle(), "quality", OIDN_QUALITY_BALANCED);
			oidnSetFilterInt(alpha_filter.getHandle(), "quality", OIDN_QUALITY_BALANCED);

			check_error();
		}

		template<typename PixelType>
		std::future<double> denoise_impl(image_type<PixelType> in, image_type<PixelType> out, roccu::gpu_event& event) {

			oidn::BufferRef in_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(in.ptr()), in.size_bytes());
			oidn::BufferRef out_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(out.ptr()), out.size_bytes());

			constexpr static auto format = std::is_same_v<PixelType, half3> ? oidn::Format::Half3 : oidn::Format::Float3;

			filter.setImage("color", in_buf, format, in.dims().x, in.dims().y);
			filter.setImage("output", out_buf, format, out.dims().x, out.dims().y);
			filter.commit();
			check_error();

			auto timer = std::make_shared<rfkt::timer>();
			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func([timer]() { timer->reset(); });

			filter.executeAsync();
			check_error();
			stream.record(event);

			stream.host_func([timer = std::move(timer), promise = std::move(promise)]() mutable {
				promise.set_value(timer->count());
			});

			return future;
		}

		template<typename PixelType>
		std::future<double> denoise_impl_alpha(image_type<PixelType> in, image_type<PixelType> out, roccu::gpu_event& event) {
			oidn::BufferRef in_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(in.ptr()), in.size_bytes());
			oidn::BufferRef out_buf = oidnNewSharedBuffer(device.getHandle(), std::bit_cast<void*>(out.ptr()), out.size_bytes());

			constexpr static auto format = std::is_same_v<PixelType, half4> ? oidn::Format::Half3 : oidn::Format::Float3;
			constexpr static auto alpha_format = std::is_same_v<PixelType, half4> ? oidn::Format::Half : oidn::Format::Float;
			constexpr static auto pixel_size = sizeof(PixelType);
			constexpr static auto channel_size = sizeof(PixelType::x);

			filter.setImage("color", in_buf, format, in.dims().x, in.dims().y, 0, pixel_size, pixel_size * in.dims().x);
			filter.setImage("output", out_buf, format, out.dims().x, out.dims().y, 0, pixel_size, pixel_size * out.dims().x);
			filter.commit();
			check_error();

			alpha_filter.setImage("color", in_buf, alpha_format, in.dims().x, in.dims().y, channel_size * 3, pixel_size, pixel_size * in.dims().x);
			alpha_filter.setImage("output", out_buf, alpha_format, out.dims().x, out.dims().y, channel_size * 3, pixel_size, pixel_size * out.dims().x);
			alpha_filter.commit();

			auto timer = std::make_shared<rfkt::timer>();
			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func([timer]() { timer->reset(); });

			filter.executeAsync();
			alpha_filter.executeAsync();
			check_error();
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
			return denoise_impl_alpha(in, out, event);
		}

		std::future<double> denoise(image_type<float3> in, image_type<float3> out, roccu::gpu_event& event) override {
			return denoise_impl(in, out, event);
		}

		std::future<double> denoise(image_type<float4> in, image_type<float4> out, roccu::gpu_event& event) override {
			return denoise_impl_alpha(in, out, event);
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
		oidn::FilterRef alpha_filter;
	};

}