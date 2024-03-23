#include <set>
#include <future>

#include <roccu.h>

#include <librefrakt/factory.h>
#include <librefrakt/gpu_buffer.h>

namespace rfkt {

	namespace denoiser_flag {
		using flags = std::uint32_t;

		constexpr static std::uint32_t none = 0;
		constexpr static std::uint32_t upscale = 1;
		constexpr static std::uint32_t tiled = 2;

	}

	struct denoiser : factory<denoiser, uint2, denoiser_flag::flags, gpu_stream&> {
		struct meta_type {
			std::string_view pretty_name;
			std::size_t priority;
			std::set<roccu_api> supported_apis;
			bool upscale_supported;
		};

		using pixel_type = half3;
		using image_type = gpu_image<pixel_type>;

		denoiser(key) {}
		virtual ~denoiser() = default;

		virtual std::future<double> denoise(const image_type& in, image_type& out, gpu_event& event) = 0;
	};

}