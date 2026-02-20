#include <librefrakt/util/cuda.hpp>
#include <stdexcept>

#include <spdlog/spdlog.h>

auto rfkt::cuda::init() -> roccu::context
{
    CUdevice dev{};
    CUcontext ctx{};

    if(auto api = roccuInit(); api == ROCCU_API_NONE) {
        throw std::runtime_error("Failed to initialize CUDA");
    }

    ROCCU_SAFE_CALL(cuInit(0));
    ROCCU_SAFE_CALL(cuDeviceGet(&dev, 0));
    ROCCU_SAFE_CALL(cuCtxCreate(&ctx, 0x01 | 0x08, dev));

    auto devobj = roccu::device_t{ dev };

    std::size_t max_persist_l2 = devobj.max_persist_l2_cache_size();

    SPDLOG_INFO("Using CUDA device: {}", devobj.name());
    SPDLOG_INFO("   Compute capability: {}.{}", devobj.compute_major(), devobj.compute_minor());
    SPDLOG_INFO("   Max threads per block: {}", devobj.max_threads_per_block());
    SPDLOG_INFO("   Max shared per block: {}", devobj.max_shared_per_block());
    SPDLOG_INFO("   Max threads per MP: {}", devobj.max_threads_per_mp());
    SPDLOG_INFO("   Max shared per MP: {}", devobj.max_shared_per_mp());
    SPDLOG_INFO("   Max blocks per MP: {}", devobj.max_blocks_per_mp());
    SPDLOG_INFO("   Warp size: {}", devobj.warp_size());
    SPDLOG_INFO("   L2 cache size: {}", devobj.l2_cache_size());

    auto max_bins_in_l2 = devobj.l2_cache_size() / 16;
    auto hist_16_9 = max_bins_in_l2 / 16 * 9;
    auto hist_4_3 = max_bins_in_l2 / 16 * 4;

    SPDLOG_INFO("   Max bins in L2: {}", max_bins_in_l2);
    SPDLOG_INFO("   Histogram 16:9 size: {}", hist_16_9);
    SPDLOG_INFO("   Histogram 4:3 size: {}", hist_4_3);

    return { ctx, dev };
}
