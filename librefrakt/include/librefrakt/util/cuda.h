#pragma once

#include <cuda.h>
#include <vector_types.h>
#include <vector>
#include <string>
#include <map>

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("`%s` failed with result: %s", #x, msg);             \
      exit(1);                                                    \
    }                                                             \
  } while(0)

namespace rfkt::cuda {

    struct execution_config {
        int grid;
        int block;
        int shared_per_block;
    };

    CUstream thread_local_stream();

    class device_t {
    public:
        device_t(CUdevice dev) : dev_(dev) {}

        auto max_threads_per_block() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK>(); }
        auto max_shared_per_block() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>(); }
        auto clock_rate() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_CLOCK_RATE>(); }
        auto mp_count() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>(); }
        auto max_threads_per_mp() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR>(); }
        auto compute_major() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(); }
        auto compute_minor() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(); }
        auto max_shared_per_mp() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR>(); }
        auto max_blocks_per_mp() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR>(); }
        auto warp_size() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_WARP_SIZE>(); }
        auto reserved_shared_per_block() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK>(); }
        bool cooperative_supported() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH>() == 1; }
        

        auto max_concurrent_threads() const noexcept { return max_threads_per_mp() * mp_count(); }
        auto max_concurrent_blocks() const noexcept { return max_blocks_per_mp() * mp_count(); }
        auto max_concurrent_warps() const noexcept { return max_concurrent_threads() / warp_size(); }
        auto max_warps_per_mp() const noexcept { return max_threads_per_mp() / warp_size(); }

        auto concurrent_block_configurations() const noexcept -> std::vector<execution_config> {
            auto ret = std::vector<execution_config>{};
            for (int i = 1; i <= max_blocks_per_mp(); i++) {
                if (max_warps_per_mp() % i == 0 && max_threads_per_mp() / i < max_threads_per_block()) {
                    ret.push_back({ i * mp_count(), max_threads_per_mp() / i, (max_shared_per_mp() - reserved_shared_per_block() * i) / i});
                }
            }

            return ret;
        }

        auto name() const noexcept {
            char buf[128];
            cuDeviceGetName(buf, sizeof(buf), dev_);
            return std::string{ buf };
        }

    private:
        CUdevice dev_;

        mutable std::map<CUdevice_attribute, int> cached_attrs;

        template<CUdevice_attribute attrib>
        int attribute() const noexcept {
            if (cached_attrs.contains(attrib)) return cached_attrs[attrib];
            int ret;
            cuDeviceGetAttribute(&ret, attrib, dev_);
            cached_attrs[attrib] = ret;
            return ret;
        }
    };

    class context {
    public:
        context(CUcontext ctx, CUdevice dev) : ctx_(ctx), dev_(dev), valid(true) {}
        context() : valid(false) {}

        operator CUcontext () { return ctx_; }
        CUcontext* ptr() { return &ctx_; }

        device_t device() const {
            return device_t{ dev_ };
        }

        static context current() {
            CUcontext ctx;
            CUdevice dev;
            cuCtxGetCurrent(&ctx);
            cuCtxGetDevice(&dev);
            return { ctx, dev };
        }

        void make_current() const {
            cuCtxSetCurrent(ctx_);
        }

        void make_current_if_not() const {
            CUcontext current;
            cuCtxGetCurrent(&current);

            if (current != ctx_) make_current();
        }

        void restart() {
            cuCtxDestroy(ctx_);
            cuCtxCreate(&ctx_, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev_);
        }

    private:
        CUcontext ctx_ = nullptr;
        CUdevice dev_ = 0;
        bool valid = false;
    };

    auto init()->context;
}