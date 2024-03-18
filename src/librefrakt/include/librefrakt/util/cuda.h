#pragma once

#include <roccu.h>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <array>

#include <librefrakt/util/filesystem.h>
#include <librefrakt/util.h>

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    RUresult result = x;                                          \
    if (result != RU_SUCCESS) {                                 \
      const char *msg;                                            \
      ruGetErrorName(result, &msg);                               \
      printf("`%s` failed with result: %s\n", #x, msg);             \
      printf("%s", rfkt::stacktrace().c_str());                   \
      __debugbreak();                                             \
      exit(1);                                                    \
    }                                                             \
  } while(0)

using float16 = std::uint16_t;

using half3 = ushort3;

namespace rfkt::cuda {

    std::optional<rfkt::fs::path> check_and_download_cudart();

    struct execution_config {
        int grid;
        int block;
        int shared_per_block;
    };

    class device_t {
    public:
        explicit(false) device_t(RUdevice dev) : dev_(dev) {}

        auto max_threads_per_block() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK>(); }
        auto max_shared_per_block() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK>(); }
        auto clock_rate() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_CLOCK_RATE>(); }
        auto mp_count() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT>(); }
        auto max_threads_per_mp() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR>(); }
        auto compute_major() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(); }
        auto compute_minor() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(); }
        auto max_shared_per_mp() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR>(); }
        auto max_blocks_per_mp() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR>(); }
        auto warp_size() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_WARP_SIZE>(); }
        auto reserved_shared_per_block() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK>(); }
        bool cooperative_supported() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH>() == 1; }
        auto l2_cache_size() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE>(); }
        auto max_persist_l2_cache_size() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE>(); }
        auto max_access_policy_window_size() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE>(); }
        auto max_registers_per_block() const noexcept { return attribute<RU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK>(); }
        
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

        std::string_view name() const noexcept {
            if(!name_.empty()) return name_;
            name_.resize(128);
            ruDeviceGetName(name_.data(), static_cast<int>(name_.size()), dev_);
            return name_;
        }

    private:
        RUdevice dev_;

        mutable std::map<RUdevice_attribute, int> cached_attrs;
        mutable std::string name_;

        template<RUdevice_attribute attrib>
        int attribute() const noexcept {
            if (cached_attrs.contains(attrib)) return cached_attrs[attrib];
            int ret;
            ruDeviceGetAttribute(&ret, attrib, dev_);
            cached_attrs[attrib] = ret;
            return ret;
        }
    };

    class context {
    public:
        context(RUcontext ctx, RUdevice dev) : ctx_(ctx), dev_(dev) {}
        context() = default;

        explicit(false) operator RUcontext () const { return ctx_; }
        RUcontext* ptr() { return &ctx_; }

        device_t device() const {
            return device_t{ dev_ };
        }

        static context current() {
            RUcontext ctx;
            RUdevice dev;
            ruCtxGetCurrent(&ctx);
            ruCtxGetDevice(&dev);
            return { ctx, dev };
        }

        void make_current() const {
            ruCtxSetCurrent(ctx_);
        }

        void make_current_if_not() const {
            RUcontext current;
            ruCtxGetCurrent(&current);

            if (current != ctx_) make_current();
        }

        void restart() {
            ruCtxDestroy(ctx_);
            ruCtxCreate(&ctx_, 0x01 | 0x08, dev_);
        }

    private:
        RUcontext ctx_ = nullptr;
        RUdevice dev_ = 0;
    };

    auto init()->context;

}

namespace rfkt {
    class gpu_stream {
    public:

        gpu_stream() noexcept {
            RUstream stream;
            int least, most;
            CUDA_SAFE_CALL(ruCtxGetStreamPriorityRange(&least, &most));
            CUDA_SAFE_CALL(ruStreamCreateWithPriority(&stream, 0x1, most));

            this->stream = stream;
        }

        ~gpu_stream() {
            if (not stream) return;
            sync();
            ruStreamDestroy(stream);
        }

        gpu_stream(const gpu_stream&) noexcept = delete;
        gpu_stream& operator=(const gpu_stream&) noexcept = delete;

        gpu_stream(gpu_stream&& o) noexcept {
            std::swap(stream, o.stream);
        }

        gpu_stream& operator=(gpu_stream&& o) noexcept {
            std::swap(stream, o.stream);
            return *this;
        }

        explicit(false) [[nodiscard]] operator RUstream() const noexcept {
            return stream;
        }

        void sync() {
            if (not stream) return;
            CUDA_SAFE_CALL(ruStreamSynchronize(stream));
        }

        using host_func_t = std::move_only_function<void(void)>;
        void host_func(host_func_t&& cb) noexcept {
            auto func = new host_func_t{ std::move(cb) };

            auto res = ruLaunchHostFunc(stream, [](void* ud) {
                auto func_p = (host_func_t*)ud;
                (*func_p)();
                delete func_p;
                }, func);

            if (res != RU_SUCCESS) {
                delete func;
            }

            CUDA_SAFE_CALL(res);
        }

    private:
        RUstream stream = nullptr;
    };
}