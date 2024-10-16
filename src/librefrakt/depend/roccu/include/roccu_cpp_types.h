#pragma once

#include <vector>
#include <string_view>
#include <map>
#include <cstdio>
#include <span>
#include <functional>
#include <bit>
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

        void wait_for(const gpu_event& ev) {
			if (not stream) return;
			ROCCU_SAFE_CALL(ruStreamWaitEvent(stream, ev, 0));
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
    public:
        using value_type = Contained;
        constexpr static bool is_owner = Ownership == buffer_ownership::owner;
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
            if constexpr(sizeof(Contained) % 4 == 0) {
				ROCCU_SAFE_CALL(ruMemsetD32(ptr_, 0, size_));
			} else if constexpr(sizeof(Contained) % 2 == 0) {
				ROCCU_SAFE_CALL(ruMemsetD16(ptr_, 0, size_));
			} else {
				ROCCU_SAFE_CALL(ruMemsetD8(ptr_, 0, size_));
			}
        }

        void clear(RUstream stream) {
            if constexpr(sizeof(Contained) % 4 == 0) {
				ROCCU_SAFE_CALL(ruMemsetD32Async(ptr_, 0, size_, stream));
			} else if constexpr(sizeof(Contained) % 2 == 0) {
				ROCCU_SAFE_CALL(ruMemsetD16Async(ptr_, 0, size_, stream));
			} else {
				ROCCU_SAFE_CALL(ruMemsetD8Async(ptr_, 0, size_, stream));
			}
        }

        void clear(Contained value) requires(element_size == 1 || element_size == 2 || element_size == 4) {
            if constexpr (element_size == 1) {
                ROCCU_SAFE_CALL(ruMemsetD8(ptr_, value, size_));
            }
            else if constexpr (element_size == 2) {
                ROCCU_SAFE_CALL(ruMemsetD16(ptr_, value, size_));
            }
            else {
                ROCCU_SAFE_CALL(ruMemsetD32(ptr_, value, size_));
            }
        }

        void clear(Contained value, RUstream stream) requires(element_size == 1 || element_size == 2 || element_size == 4) {
			if constexpr (element_size == 1) {
				ROCCU_SAFE_CALL(ruMemsetD8Async(ptr_, value, size_, stream));
			}
			else if constexpr (element_size == 2) {
				ROCCU_SAFE_CALL(ruMemsetD16Async(ptr_, value, size_, stream));
			}
			else {
				ROCCU_SAFE_CALL(ruMemsetD32Async(ptr_, value, size_, stream));
			}
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

    template<typename PixelType, buffer_ownership Ownership>
    class gpu_image_base {
    public:
        using pixel_type = PixelType;
        constexpr static bool is_owner = Ownership == buffer_ownership::owner;
        constexpr static std::size_t element_size = sizeof(PixelType);

        constexpr gpu_image_base() noexcept = default;

        explicit gpu_image_base(uint2 dims) requires is_owner : dims_(dims), pitch_(dims_.x) {
			ROCCU_SAFE_CALL(ruMemAlloc(&ptr_, size_bytes()));
		}

        gpu_image_base(uint2 dims, RUstream stream) requires is_owner : dims_(dims), pitch_(dims_.x) {
			ROCCU_SAFE_CALL(ruMemAllocAsync(&ptr_, size_bytes(), stream));
		}

        gpu_image_base(std::size_t width, std::size_t height) requires is_owner : dims_(width, height), pitch_(dims_.x) {
			ROCCU_SAFE_CALL(ruMemAlloc(&ptr_, size_bytes()));
		}

        gpu_image_base(std::size_t width, std::size_t height, RUstream stream) requires is_owner : dims_(width, height), pitch_(dims_.x) {
            ROCCU_SAFE_CALL(ruMemAllocAsync(&ptr_, size_bytes(), stream));
        }

        gpu_image_base(RUdeviceptr ptr, uint2 dims, std::size_t pitch) requires !is_owner : ptr_(ptr), dims_(dims), pitch_(pitch) {}

        ~gpu_image_base() {
			if constexpr (!is_owner) return;
			if (ptr_) {
				ROCCU_SAFE_CALL(ruMemFree(ptr_));
			}
		}

		gpu_image_base(const gpu_image_base&) requires is_owner = delete;
		gpu_image_base& operator=(const gpu_image_base&) requires is_owner = delete;

		gpu_image_base(const gpu_image_base&) requires !is_owner = default;
		gpu_image_base& operator=(const gpu_image_base&) requires !is_owner = default;

        gpu_image_base(gpu_image_base&& o) noexcept {
            std::swap(ptr_, o.ptr_);
            std::swap(dims_, o.dims_);
            std::swap(pitch_, o.pitch_);
        }

        gpu_image_base& operator=(gpu_image_base&& o) noexcept {
			std::swap(ptr_, o.ptr_);
            std::swap(dims_, o.dims_);
			std::swap(pitch_, o.pitch_);
			return *this;
		}

        explicit(false) operator gpu_image_base<PixelType, buffer_ownership::view>() requires is_owner {
            return { ptr_, dims_, pitch_ };
        }

        constexpr auto ptr() const noexcept { return ptr_; }
        constexpr auto width() const noexcept { return dims_.x; }
        constexpr auto height() const noexcept { return dims_.y; }
        constexpr auto dims() const noexcept { return dims_; }
        constexpr auto area() const noexcept { return dims_.x * dims_.y; }
        constexpr auto pitch() const noexcept { return pitch_; }
        constexpr auto size_bytes() const noexcept { return area() * element_size; }

        constexpr bool valid() const noexcept { return ptr_ != 0; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        void clear() requires is_owner {
            ROCCU_SAFE_CALL(ruMemsetD8(ptr_, 0, size_bytes()));
        }

        void clear(RUstream stream) requires is_owner {
            ROCCU_SAFE_CALL(ruMemsetD8Async(ptr_, 0, size_bytes(), stream));
        }

        auto to_host(std::span<PixelType> dest_host) const {
            auto copy_param = to_host_memcpy(dest_host);
            ROCCU_SAFE_CALL(ruMemcpy2D(&copy_param));
        }

        auto to_host(std::span<PixelType> dest_host, RUstream stream) const {
            auto copy_param = to_host_memcpy(dest_host);
			ROCCU_SAFE_CALL(ruMemcpy2DAsync(&copy_param, stream));
		}

        auto to_host() const -> std::vector<PixelType> {
            auto ret = std::vector<PixelType>(area());
            to_host(ret);
            return ret;
        }

        auto to_host(RUstream stream) const -> std::vector<PixelType>  {
			auto ret = std::vector<PixelType>(area());
			to_host(ret, stream);
			return ret;
		}

        void from_host(std::span<const PixelType> src_host)  {
            auto copy_param = from_host_memcpy(src_host);
			ROCCU_SAFE_CALL(ruMemcpy2D(&copy_param));
		}

        void from_host(std::span<const PixelType> src_host, RUstream stream) {
            auto copy_param = from_host_memcpy(src_host);
            ROCCU_SAFE_CALL(ruMemcpy2DAsync(&copy_param, stream));
        }

        void free_async(RUstream stream) requires is_owner {
			if (ptr_) {
				ROCCU_SAFE_CALL(ruMemFreeAsync(ptr_, stream));
			}

			ptr_ = 0;
            dims_ = { 0, 0 };
			pitch_ = 0;
		}

        auto sub_image(uint2 offset, uint2 dims) -> gpu_image_base<PixelType, buffer_ownership::view> {
            auto new_ptr = ptr_ + element_size * (offset.y * pitch_ + offset.x);
			return { new_ptr, dims, pitch_ };
		}

    private:

        auto to_host_memcpy(std::span<PixelType> dest_host) const {
            RU_MEMCPY2D copy;
            copy.srcMemoryType = RU_MEMORYTYPE_DEVICE;
            copy.srcDevice = ptr_;
            copy.srcPitch = pitch_ * element_size;
            copy.srcXInBytes = 0;
            copy.srcY = 0;

            copy.dstMemoryType = RU_MEMORYTYPE_HOST;
            copy.dstHost = dest_host.data();
            copy.dstPitch = dims_.x * element_size;
            copy.dstXInBytes = 0;
            copy.dstY = 0;

            copy.WidthInBytes = dims_.x * element_size;
            copy.Height = dims_.y;

            return copy;
		}
        
        auto from_host_memcpy(std::span<const PixelType> src_host) const {
			RU_MEMCPY2D copy;
			copy.srcMemoryType = RU_MEMORYTYPE_HOST;
			copy.srcHost = src_host.data();
			copy.srcPitch = dims_.x * element_size;
			copy.srcXInBytes = 0;
			copy.srcY = 0;

			copy.dstMemoryType = RU_MEMORYTYPE_DEVICE;
			copy.dstDevice = ptr_;
			copy.dstPitch = pitch_ * element_size;
			copy.dstXInBytes = 0;
			copy.dstY = 0;

			copy.WidthInBytes = dims_.x * element_size;
			copy.Height = dims_.y;

			return copy;
		}

        RUdeviceptr ptr_ = 0;
        uint2 dims_ = { 0, 0 };
        std::size_t pitch_ = 0;
    };

    template<typename PixelType>
    using gpu_image = gpu_image_base<PixelType, buffer_ownership::owner>;

    template<typename PixelType>
    using gpu_image_view = gpu_image_base<PixelType, buffer_ownership::view>;
}