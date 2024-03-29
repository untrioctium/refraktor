#pragma once

#include <vector>
#include <string_view>
#include <map>
#include <cstdio>
#include <span>
#include <functional>

#include <roccu.h>

#define ROCCU_SAFE_CALL(x)                                         \
  do {                                                            \
    RUresult result = x;                                          \
    if (result != RU_SUCCESS) {                                 \
      const char *msg;                                            \
      ruGetErrorName(result, &msg);                               \
      printf("`%s` failed with result: %s\n", #x, msg);             \
      __debugbreak();                                             \
      exit(1);                                                    \
    }                                                             \
  } while(0)

namespace roccu {

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
                    ret.push_back({ i * mp_count(), max_threads_per_mp() / i, (max_shared_per_mp()) / i - reserved_per_block });
                }
            }

            return ret;
        }

        std::string_view name() const noexcept {
            if (!name_.empty()) return name_;
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

    class gpu_event {
    public:
        gpu_event() noexcept {
            ROCCU_SAFE_CALL(ruEventCreate(&event, 0));
        };

        ~gpu_event() {
            if (not event) return;
            ruEventDestroy(event);
        }

        gpu_event(const gpu_event&) = delete;
        gpu_event& operator=(const gpu_event&) = delete;

        gpu_event(gpu_event&& o) noexcept {
            std::swap(event, o.event);
        }

        gpu_event& operator=(gpu_event&& o) noexcept {
            std::swap(event, o.event);
            return *this;
        }

        explicit(false) [[nodiscard]] operator RUevent() const noexcept {
            return event;
        }

        void sync() {
            if (not event) return;
            ROCCU_SAFE_CALL(ruEventSynchronize(event));
        }

        float elapsed_time(const gpu_event& end) {
            float ms;
            ROCCU_SAFE_CALL(ruEventElapsedTime(&ms, event, end));
            return ms;
        }

    private:
        RUevent event = nullptr;
    };

    struct [[nodiscard]] l2_persister {
        l2_persister(RUdeviceptr ptr, std::size_t size, float ratio, RUstream stream) :
            ptr_(ptr),
            stream_(stream) {

            RUlaunchAttributeValue attr;
            attr.accessPolicyWindow.basePtr = ptr_;
            attr.accessPolicyWindow.numBytes = size;
            attr.accessPolicyWindow.hitRatio = ratio;
            attr.accessPolicyWindow.hitProp = RU_ACCESS_PROPERTY_PERSISTING;
            attr.accessPolicyWindow.missProp = RU_ACCESS_PROPERTY_STREAMING;

            ROCCU_SAFE_CALL(ruStreamSetAttribute(stream_, RU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
        }

        ~l2_persister() {
            RUlaunchAttributeValue attr;
            attr.accessPolicyWindow.basePtr = ptr_;
            attr.accessPolicyWindow.numBytes = 0;
            attr.accessPolicyWindow.hitRatio = 0.0f;
            attr.accessPolicyWindow.hitProp = RU_ACCESS_PROPERTY_NORMAL;
            attr.accessPolicyWindow.missProp = RU_ACCESS_PROPERTY_NORMAL;

            ROCCU_SAFE_CALL(ruStreamSetAttribute(stream_, RU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
        }

    private:
        RUdeviceptr ptr_ = 0;
        RUstream stream_ = 0;
    };

    enum class buffer_ownership {
        owner,
        view
    };

    class gpu_stream {
    public:

        gpu_stream() noexcept {
            RUstream stream;
            int least, most;
            ROCCU_SAFE_CALL(ruCtxGetStreamPriorityRange(&least, &most));
            ROCCU_SAFE_CALL(ruStreamCreateWithPriority(&stream, 0x1, most));

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
            ROCCU_SAFE_CALL(ruStreamSynchronize(stream));
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

            ROCCU_SAFE_CALL(res);
        }

        void record(gpu_event& ev) {
            if (not stream) return;
            ROCCU_SAFE_CALL(ruEventRecord(ev, stream));
        }

    private:
        RUstream stream = nullptr;
    };

    template<class Contained, buffer_ownership Ownership>
    class gpu_buffer_base {
    private:
        constexpr static bool is_owner = Ownership == buffer_ownership::owner;

    public:
        constexpr static std::size_t element_size = sizeof(Contained);

        constexpr gpu_buffer_base() noexcept = default;

        explicit gpu_buffer_base(std::size_t size) requires is_owner : size_(size) {
            ROCCU_SAFE_CALL(ruMemAlloc(&ptr_, size_bytes()));
		}

        gpu_buffer_base(std::size_t size, RUstream stream) requires is_owner : size_(size) {
            ROCCU_SAFE_CALL(ruMemAllocAsync(&ptr_, size_bytes(), stream));
		}

        gpu_buffer_base(RUdeviceptr ptr, std::size_t size) requires !is_owner : ptr_(ptr), size_(size) {}
        //explicit(false) gpu_buffer_base(const gpu_buffer_base<Contained, buffer_ownership::owner>& o) requires !is_owner  : ptr_(o.ptr()), size_(o.size()) {}

        explicit(false) operator gpu_buffer_base<Contained, buffer_ownership::view>() const requires is_owner {
			return { ptr_, size_ };
		}

        ~gpu_buffer_base() {
            if constexpr (!is_owner) return;
            if (ptr_) {
                ROCCU_SAFE_CALL(ruMemFree(ptr_));
			}
		}

        gpu_buffer_base(const gpu_buffer_base&) requires is_owner  = delete;
        gpu_buffer_base& operator=(const gpu_buffer_base&) requires is_owner  = delete;

        gpu_buffer_base(const gpu_buffer_base&) requires !is_owner  = default;
        gpu_buffer_base& operator=(const gpu_buffer_base&) requires !is_owner  = default;

        gpu_buffer_base(gpu_buffer_base&& o) noexcept {
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
		}

        gpu_buffer_base& operator=(gpu_buffer_base&& o) noexcept {
            std::swap(ptr_, o.ptr_);
            std::swap(size_, o.size_);
            return *this;
        }

        constexpr auto ptr() const noexcept { return ptr_; }
        constexpr auto size() const noexcept { return size_; }
        constexpr auto size_bytes() const noexcept { return size_ * element_size; }

        constexpr bool valid() const noexcept { return ptr_ != 0; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        void clear() {
            ROCCU_SAFE_CALL(ruMemsetD8(ptr_, 0, size_bytes()));
        }

        void clear(RUstream stream) {
            ROCCU_SAFE_CALL(ruMemsetD8Async(ptr_, 0, size_bytes(), stream));
        }

        void to_host(std::span<Contained> dest_host) const {
            if (size_ == 0) return;
            ROCCU_SAFE_CALL(ruMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
        }

        void to_host(std::span<Contained> dest_host, RUstream stream) const {
			if (size_ == 0) return;
            ROCCU_SAFE_CALL(ruMemcpyDtoHAsync(dest_host.data(), ptr_, min_size(dest_host.size()), stream));
		}

        auto to_host() const -> std::vector<Contained> {
			if (size_ == 0) return {};
			auto ret = std::vector<Contained>(size_);
			to_host(ret);
			return ret;
		}

        auto to_host(RUstream stream) const -> std::vector<Contained> {
            if (size_ == 0) return {};
            auto ret = std::vector<Contained>(size_);
            to_host(ret, stream);
            return ret;
        }

        void from_host(std::span<const Contained> src_host) {
			if (size_ == 0) return;
            ROCCU_SAFE_CALL(ruMemcpyHtoD(ptr_, src_host.data(), min_size(src_host.size())));
		}

        void from_host(std::span<const Contained> src_host, RUstream stream) {
            if (size_ == 0) return;
            ROCCU_SAFE_CALL(ruMemcpyHtoDAsync(ptr_, src_host.data(), min_size(src_host.size()), stream));
        }

        void free_async(RUstream stream) requires is_owner {
            if (ptr_) {
                ROCCU_SAFE_CALL(ruMemFreeAsync(ptr_, stream));
            }

            ptr_ = 0;
            size_ = 0;
        }

    private:

        constexpr auto min_size(std::size_t other_size) const noexcept {
            using namespace std;
            return min(other_size, size_) * sizeof(Contained);
        }

        RUdeviceptr ptr_ = 0;
        std::size_t size_ = 0;
    };

    template<typename Contained = std::byte>
    using gpu_buffer = gpu_buffer_base<Contained, buffer_ownership::owner>;

    template<typename Contained = std::byte>
    using gpu_span = gpu_buffer_base<Contained, buffer_ownership::view>;
}