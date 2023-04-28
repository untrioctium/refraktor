#include <cuda.h>
#include <vector_types.h>
#include <memory>
#include <future>

#include <librefrakt/cuda_buffer.h>

namespace rfkt {
    class denoiser {
    public:

        static void init(CUcontext ctx);

        denoiser(uint2 max_dims, bool upscale_2x);

        denoiser(const denoiser&) = delete;
        denoiser& operator=(const denoiser&) = delete;

        denoiser(denoiser&& d) noexcept;
        denoiser& operator=(denoiser&& d) noexcept;

        ~denoiser();

        std::future<double> denoise(uint2 dims, cuda_span<half3> in, cuda_span<half3> out, cuda_stream& stream);

        static double benchmark(uint2 dims, bool upscale, std::uint32_t num_passes, cuda_stream& stream);

    private:
        class denoiser_impl;
        std::unique_ptr<denoiser_impl> impl;
    };
}

