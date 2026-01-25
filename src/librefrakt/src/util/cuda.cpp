#include <librefrakt/util/cuda.hpp>

auto rfkt::cuda::init() -> roccu::context
{
    RUdevice dev;
    RUcontext ctx;

    roccuInit();

    ROCCU_SAFE_CALL(ruInit(0));
    ROCCU_SAFE_CALL(ruDeviceGet(&dev, 0));
    ROCCU_SAFE_CALL(ruCtxCreate(&ctx, 0x01 | 0x08, dev));

    auto devobj = roccu::device_t{ dev };

    std::size_t max_persist_l2 = devobj.max_persist_l2_cache_size();
    ROCCU_SAFE_CALL(ruCtxSetLimit(RU_LIMIT_PERSISTING_L2_CACHE_SIZE, max_persist_l2));

    return { ctx, dev };
}
