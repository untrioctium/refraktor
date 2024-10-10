#include <set>
#include <future>

#include <roccu.h>

#include <librefrakt/factory.h>
#include <librefrakt/gpu_buffer.h>
#include <librefrakt/util/cuda.h>

namespace rfkt {

	namespace denoiser_flag {
		using flags = std::uint32_t;

		constexpr static std::uint32_t none = 0;
		constexpr static std::uint32_t upscale = 1;
		constexpr static std::uint32_t tiled = 2;

	}

	struct denoiser : factory<denoiser, uint2, denoiser_flag::flags, roccu::gpu_stream&> {
		struct meta_type {
			std::string_view pretty_name;
			std::size_t priority;
			std::set<roccu_api> supported_apis;
			bool upscale_supported;
		};

		denoiser(key) {}
		virtual ~denoiser() = default;

		template<typename PixelType>
		using image_type = roccu::gpu_image_view<PixelType>;

		virtual std::future<double> denoise(image_type<half3> in, image_type<half3> out, roccu::gpu_event& stream) = 0;
		virtual std::future<double> denoise(image_type<half4> in, image_type<half4> out, roccu::gpu_event& stream) = 0;
		virtual std::future<double> denoise(image_type<float3> in, image_type<float3> out, roccu::gpu_event& stream) = 0;
		virtual std::future<double> denoise(image_type<float4> in, image_type<float4> out, roccu::gpu_event& stream) = 0;
	};

}