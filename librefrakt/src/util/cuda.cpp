#include <librefrakt/util/cuda.h>

CUstream rfkt::cuda::thread_local_stream()
{
    struct stream_wrapper {
        CUstream stream = nullptr;

        stream_wrapper() {
            int least, most;
            CUDA_SAFE_CALL(cuCtxGetStreamPriorityRange(&least, &most));
            CUDA_SAFE_CALL(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, most));
        }

        ~stream_wrapper() {
            if (stream != nullptr) cuStreamDestroy(stream);
        }
    };

    thread_local auto stream = stream_wrapper{};
    return stream.stream;
}

auto rfkt::cuda::init() -> context
{
    CUdevice dev;
    CUcontext ctx;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&dev, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev));
    return { ctx, dev };
}
