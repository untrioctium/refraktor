#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <cuda.h>
#include <vector_types.h>
#include <memory>
#include <future>

#include <ezrtc.h>

#include <librefrakt/cuda_buffer.h>

namespace rfkt {

    namespace denoiser_flag {
        using flags = std::uint32_t;

        constexpr static std::uint32_t none = 0;
        constexpr static std::uint32_t upscale = 1;
        constexpr static std::uint32_t tiled = 2;

    }

    class denoiser {
    public:
        static void init(CUcontext ctx);

        denoiser(uint2 max_dims, denoiser_flag::flags options = denoiser_flag::none);

        denoiser(const denoiser&) = delete;
        denoiser& operator=(const denoiser&) = delete;

        denoiser(denoiser&& d) noexcept;
        denoiser& operator=(denoiser&& d) noexcept;

        ~denoiser();

        using pixel_type = half3;
        using image_type = cuda_image<pixel_type>;
        std::future<double> denoise(const image_type& in, image_type& out, cuda_stream& stream);

        static double benchmark(uint2 dims, denoiser_flag::flags options, std::uint32_t num_passes, cuda_stream& stream);

    private:
        class denoiser_impl;
        std::unique_ptr<denoiser_impl> impl;
    };

    class hdr_denoiser {
    public:
        static void init(CUcontext ctx);

        hdr_denoiser(ezrtc::compiler& kc, uint2 max_dims, denoiser_flag::flags options = denoiser_flag::none);

        hdr_denoiser(const hdr_denoiser&) = delete;
        hdr_denoiser& operator=(const hdr_denoiser&) = delete;

        hdr_denoiser(hdr_denoiser&& d) noexcept;
        hdr_denoiser& operator=(hdr_denoiser&& d) noexcept;

        ~hdr_denoiser();

        using pixel_type = float4;
        using image_type = cuda_image<pixel_type>;
        std::future<double> denoise(const image_type& in, image_type& out, cuda_stream& stream);

        static double benchmark(uint2 dims, denoiser_flag::flags options, std::uint32_t num_passes, cuda_stream& stream);

    private:
        class denoiser_impl;
        std::unique_ptr<denoiser_impl> impl;

        ezrtc::cuda_module splitter;
        int block_size;
    };
}

