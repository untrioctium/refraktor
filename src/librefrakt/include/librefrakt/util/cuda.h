#pragma once

#include <cuda.h>
#include <vector_types.h>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <array>

#include <librefrakt/util/filesystem.h>

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("`%s` failed with result: %s", #x, msg);             \
      __debugbreak();                                             \
      exit(1);                                                    \
    }                                                             \
  } while(0)

using float16 = std::uint16_t;

class half3 {
    std::array<float16, 3> data;
};

namespace rfkt::cuda {

    std::optional<rfkt::fs::path> check_and_download_cudart();

    struct execution_config {
        int grid;
        int block;
        int shared_per_block;
    };

    class device_t {
    public:
        explicit(false) device_t(CUdevice dev) : dev_(dev) {}

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
        auto l2_cache_size() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE>(); }
        auto max_persist_l2_cache_size() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE>(); }
        

        auto max_concurrent_threads() const noexcept { return max_threads_per_mp() * mp_count(); }
        auto max_concurrent_blocks() const noexcept { return max_blocks_per_mp() * mp_count(); }
        auto max_concurrent_warps() const noexcept { return max_concurrent_threads() / warp_size(); }
        auto max_warps_per_mp() const noexcept { return max_threads_per_mp() / warp_size(); }

        auto concurrent_block_configurations() const noexcept -> std::vector<execution_config> {
            auto ret = std::vector<execution_config>{};
            auto reserved_per_block = reserved_shared_per_block();
            for (int i = 1; i <= max_blocks_per_mp(); i++) {
                if (max_warps_per_mp() % i == 0 && max_threads_per_mp() / i < max_threads_per_block()) {
                    ret.push_back({ i * mp_count(), max_threads_per_mp() / i, (max_shared_per_mp()) / i - reserved_per_block});
                }
            }

            return ret;
        }

        auto name() const noexcept {
            std::string ret;
            ret.resize(128);
            cuDeviceGetName(ret.data(), static_cast<int>(ret.size()), dev_);
            return ret;
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
        context(CUcontext ctx, CUdevice dev) : ctx_(ctx), dev_(dev) {}
        context() = default;

        explicit(false) operator CUcontext () const { return ctx_; }
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
    };

    auto init()->context;

}

namespace rfkt {
    class cuda_stream {
    public:

        cuda_stream() noexcept {
            CUstream stream;
            int least, most;
            CUDA_SAFE_CALL(cuCtxGetStreamPriorityRange(&least, &most));
            CUDA_SAFE_CALL(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, most));

            this->stream = stream;
        }

        ~cuda_stream() {
            if (not stream) return;
            sync();
            cuStreamDestroy(stream);
        }

        cuda_stream(const cuda_stream&) noexcept = delete;
        cuda_stream& operator=(const cuda_stream&) noexcept = delete;

        cuda_stream(cuda_stream&& o) noexcept {
            std::swap(stream, o.stream);
        }

        cuda_stream& operator=(cuda_stream&& o) noexcept {
            std::swap(stream, o.stream);
            return *this;
        }

        explicit(false) [[nodiscard]] operator CUstream() const noexcept {
            return stream;
        }

        void sync() {
            if (not stream) return;
            CUDA_SAFE_CALL(cuStreamSynchronize(stream));
        }

        using host_func_t = std::move_only_function<void(void)>;
        void host_func(host_func_t&& cb) noexcept {
            auto func = new host_func_t{ std::move(cb) };

            auto res = cuLaunchHostFunc(stream, [](void* ud) {
                auto func_p = (host_func_t*)ud;
                (*func_p)();
                delete func_p;
                }, func);

            if (res != CUDA_SUCCESS) {
                delete func;
            }

            CUDA_SAFE_CALL(res);
        }

    private:
        CUstream stream = nullptr;
    };
}