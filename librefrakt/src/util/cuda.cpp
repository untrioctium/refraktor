#include <librefrakt/util/cuda.h>

auto rfkt::cuda::init() -> context
{
    CUdevice dev;
    CUcontext ctx;

    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev);
    return { ctx, dev };
}
