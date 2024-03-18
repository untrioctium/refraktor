#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <roccu.h>
#include <memory>
#include <future>

#include <librefrakt/gpu_buffer.h>

namespace rfkt {

    namespace denoiser_flag {
        using flags = std::uint32_t;

        constexpr static std::uint32_t none = 0;
        constexpr static std::uint32_t upscale = 1;
        constexpr static std::uint32_t tiled = 2;

    }

    class denoiser {
    public:


        static void init(RUcontext ctx);

        denoiser(uint2 max_dims, denoiser_flag::flags options = denoiser_flag::none);

        denoiser(const denoiser&) = delete;
        denoiser& operator=(const denoiser&) = delete;

        denoiser(denoiser&& d) noexcept;
        denoiser& operator=(denoiser&& d) noexcept;

        ~denoiser();

        using pixel_type = half3;
        using image_type = gpu_image<pixel_type>;
        std::future<double> denoise(const image_type& in, image_type& out, gpu_stream& stream);

        static double benchmark(uint2 dims, denoiser_flag::flags options, std::uint32_t num_passes, gpu_stream& stream);

    private:
        class denoiser_impl;
        std::unique_ptr<denoiser_impl> impl;
    };
}

