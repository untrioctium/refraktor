#pragma once

#include <vector>
#include <string_view>
#include <map>
#include <cstdio>
#include <span>
#include <functional>
#include <string>
#include <roccu.hpp>
#include <cstring>

namespace roccu {

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
        auto max_access_policy_window_size() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE>(); }
        auto max_registers_per_block() const noexcept { return attribute<CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK>(); }

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
            constexpr static auto initial_name_size = 128;
            name_.resize(initial_name_size);
            cuDeviceGetName(name_.data(), static_cast<int>(name_.size()), dev_);
            name_.resize(std::strlen(name_.data()));
            return name_;
        }

    private:
        CUdevice dev_;

        mutable std::map<CUdevice_attribute, int> cached_attrs;
        mutable std::string name_;

        template<CUdevice_attribute attrib>
        int attribute() const noexcept {
            if (cached_attrs.contains(attrib)) return cached_attrs[attrib];
            int ret{};
            cuDeviceGetAttribute(&ret, attrib, dev_);
            cached_attrs[attrib] = ret;
            return ret;
        }
    };

    class context_view {
    public:
        context_view(CUcontext ctx, CUdevice dev) : ctx_(ctx), dev_(dev) {}
        context_view() = default;

        explicit(false) operator CUcontext () const { return ctx_; }
        explicit(false) operator bool() const { return ctx_ != nullptr; }

        device_t device() const {
            return device_t{ dev_ };
        }

        static context_view current() {
            CUcontext ctx{};
            CUdevice dev{};
            cuCtxGetCurrent(&ctx);
            cuCtxGetDevice(&dev);
            return { ctx, dev };
        }

        void make_current() const {
            cuCtxSetCurrent(ctx_);
        }

        void make_current_if_not() const {
            CUcontext current{};
            cuCtxGetCurrent(&current);

            if (current != ctx_) make_current();
        }

        bool operator==(const context_view& o) const noexcept { return ctx_ == o.ctx_; }

    private:
        CUcontext ctx_ = nullptr;
        CUdevice dev_ = 0;
    };

    class context {
    public:
        context(CUdevice dev) : dev_(dev) {
            ROCCU_SAFE_CALL(cuCtxCreate(&ctx_, nullptr, 0x01 | 0x08, dev_));
        }

        context(const context&) = delete;
        context& operator=(const context&) = delete;
        context(context&& o) noexcept {
            std::swap(ctx_, o.ctx_);
            std::swap(dev_, o.dev_);
        }
        context& operator=(context&& o) noexcept {
            std::swap(ctx_, o.ctx_);
            std::swap(dev_, o.dev_);
            return *this;
        }

        ~context() {
            if (ctx_) cuCtxDestroy(ctx_);
        }

        context_view view() const {
            return { ctx_, dev_ };
        }

        void make_current() const {
            ROCCU_SAFE_CALL(cuCtxSetCurrent(ctx_));
        }

        void make_current_if_not() const {
            CUcontext current{};
            ROCCU_SAFE_CALL(cuCtxGetCurrent(&current));
            if (current != ctx_) make_current();
        }

        bool operator==(const context& o) const { return ctx_ == o.ctx_; }

        device_t device() const { return device_t{ dev_ }; }

        
    private:
        CUcontext ctx_ = nullptr;
        CUdevice dev_ = 0;
    };

    struct context_scope {
        context_scope(context_view ctx) {
            cuCtxPushCurrent(ctx);
        }

        ~context_scope() {
            CUcontext current{};
            cuCtxPopCurrent(&current);
        }

        context_scope(context_scope&&) = delete;
        context_scope& operator=(context_scope&&) = delete;
        context_scope(const context_scope&) = delete;
        context_scope& operator=(const context_scope&) = delete;
    };

    class gpu_event {
    public:
        gpu_event() {
            ROCCU_SAFE_CALL(cuEventCreate(&event, 0));
        };

        ~gpu_event() {
            if (not event) return;
            cuEventDestroy(event);
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

        explicit(false) operator CUevent() const noexcept {
            return event;
        }

        void sync() {
            if (not event) return;
            ROCCU_SAFE_CALL(cuEventSynchronize(event));
        }

        float elapsed_time(const gpu_event& end) {
            float ms{};
            ROCCU_SAFE_CALL(cuEventElapsedTime(&ms, event, end));
            return ms;
        }

    private:
        CUevent event = nullptr;
    };

    //NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    struct [[nodiscard]] l2_persister {
        l2_persister(CUdeviceptr ptr, std::size_t size, float ratio, CUstream stream) :
            ptr_(ptr),
            stream_(stream) {

            auto dev_max = static_cast<std::size_t>(context_view::current().device().max_access_policy_window_size());
            size = (std::min)(size, dev_max);
            CUlaunchAttributeValue attr{};
            attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr_);
            attr.accessPolicyWindow.num_bytes = size;
            attr.accessPolicyWindow.hitRatio = ratio;
            attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
            attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_NORMAL;

            ROCCU_SAFE_CALL(cuStreamSetAttribute(stream_, CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr));
        }

        ~l2_persister() {
            CUlaunchAttributeValue attr{};
            attr.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr_);
            attr.accessPolicyWindow.num_bytes = 0;
            attr.accessPolicyWindow.hitRatio = 0.0f;
            attr.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_NORMAL;
            attr.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_NORMAL;

            // TODO: Should this failure be silent or checked?
            cuStreamSetAttribute(stream_, CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr);
        }

        l2_persister() = delete;
        l2_persister(const l2_persister&) = delete;
        l2_persister& operator=(const l2_persister&) = delete;
        l2_persister(l2_persister&&) = delete;
        l2_persister& operator=(l2_persister&&) = delete;

    private:
        CUdeviceptr ptr_ = 0;
        CUstream stream_ = 0;
    };
    //NOLINTEND(cppcoreguidelines-pro-type-union-access)

    enum class buffer_ownership {
        owner,
        view
    };

    class gpu_stream {
    public:

        gpu_stream() {
            CUstream stream{};
            int least{}, most{};
            ROCCU_SAFE_CALL(cuCtxGetStreamPriorityRange(&least, &most));
            ROCCU_SAFE_CALL(cuStreamCreateWithPriority(&stream, 0x1, most));

            this->stream = stream;
        }

        ~gpu_stream() {
            if (not stream) return;
            try { sync(); } catch (...) {}
            cuStreamDestroy(stream);
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

        explicit(false) operator CUstream() const noexcept {
            return stream;
        }

        void sync() {
            if (not stream) return;
            ROCCU_SAFE_CALL(cuStreamSynchronize(stream));
        }

        void wait_for(const gpu_event& ev) {
			if (not stream) return;
			ROCCU_SAFE_CALL(cuStreamWaitEvent(stream, ev, 0));
		}

        using host_func_t = std::move_only_function<void(void)>;
        void host_func(host_func_t&& cb) {
            auto func = new host_func_t{ std::move(cb) }; // NOLINT(cppcoreguidelines-owning-memory)

            auto res = cuLaunchHostFunc(stream, [](void* ud) {
                auto func_p = static_cast<host_func_t*>(ud);
                (*func_p)();
                delete func_p; // NOLINT(cppcoreguidelines-owning-memory)
                }, func);

            if (res != CUDA_SUCCESS) {
                delete func; // NOLINT(cppcoreguidelines-owning-memory)
            }

            ROCCU_SAFE_CALL(res);
        }

        void record(gpu_event& ev) {
            if (not stream) return;
            ROCCU_SAFE_CALL(cuEventRecord(ev, stream));
        }

    private:
        CUstream stream = nullptr;
    };

    template<class Contained, buffer_ownership Ownership>
    class gpu_buffer_base {
    public:
        using value_type = Contained;
        constexpr static bool is_owner = Ownership == buffer_ownership::owner;
        constexpr static std::size_t element_size = sizeof(Contained);

        constexpr gpu_buffer_base() noexcept = default;

        explicit gpu_buffer_base(std::size_t size) requires (is_owner) : size_(size) {
            ROCCU_SAFE_CALL(cuMemAlloc(&ptr_, size_bytes()));
		}

        gpu_buffer_base(std::size_t size, CUstream stream) requires (is_owner) : size_(size) {
            ROCCU_SAFE_CALL(cuMemAllocAsync(&ptr_, size_bytes(), stream));
		}

        gpu_buffer_base(CUdeviceptr ptr, std::size_t size) requires (!is_owner) : ptr_(ptr), size_(size) {}
        //explicit(false) gpu_buffer_base(const gpu_buffer_base<Contained, buffer_ownership::owner>& o) requires !is_owner  : ptr_(o.ptr()), size_(o.size()) {}

        explicit(false) operator gpu_buffer_base<Contained, buffer_ownership::view>() const requires (is_owner) {
			return { ptr_, size_ };
		}

        ~gpu_buffer_base() {
            if constexpr (!is_owner) return;
            if (ptr_) {
                cuMemFree(ptr_);
			}
		}

        gpu_buffer_base(const gpu_buffer_base&) requires (is_owner)  = delete;
        gpu_buffer_base& operator=(const gpu_buffer_base&) requires (is_owner) = delete;

        gpu_buffer_base(const gpu_buffer_base&) requires (!is_owner)  = default;
        gpu_buffer_base& operator=(const gpu_buffer_base&) requires (!is_owner) = default;

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
				ROCCU_SAFE_CALL(cuMemsetD32(ptr_, 0, size_));
			} else if constexpr(sizeof(Contained) % 2 == 0) {
				ROCCU_SAFE_CALL(cuMemsetD16(ptr_, 0, size_));
			} else {
				ROCCU_SAFE_CALL(cuMemsetD8(ptr_, 0, size_));
			}
        }

        void clear(CUstream stream) {
            if constexpr(sizeof(Contained) % 4 == 0) {
				ROCCU_SAFE_CALL(cuMemsetD32Async(ptr_, 0, size_, stream));
			} else if constexpr(sizeof(Contained) % 2 == 0) {
				ROCCU_SAFE_CALL(cuMemsetD16Async(ptr_, 0, size_, stream));
			} else {
				ROCCU_SAFE_CALL(cuMemsetD8Async(ptr_, 0, size_, stream));
			}
        }

        void clear(Contained value) requires((element_size == 1 || element_size == 2 || element_size == 4)) {
            if constexpr (element_size == 1) {
                ROCCU_SAFE_CALL(cuMemsetD8(ptr_, value, size_));
            }
            else if constexpr (element_size == 2) {
                ROCCU_SAFE_CALL(cuMemsetD16(ptr_, value, size_));
            }
            else {
                ROCCU_SAFE_CALL(cuMemsetD32(ptr_, value, size_));
            }
        }

        void clear(Contained value, CUstream stream) requires((element_size == 1 || element_size == 2 || element_size == 4) ) {
			if constexpr (element_size == 1) {
				ROCCU_SAFE_CALL(cuMemsetD8Async(ptr_, value, size_, stream));
			}
			else if constexpr (element_size == 2) {
				ROCCU_SAFE_CALL(cuMemsetD16Async(ptr_, value, size_, stream));
			}
			else {
				ROCCU_SAFE_CALL(cuMemsetD32Async(ptr_, value, size_, stream));
			}
		}

        void to_host(std::span<Contained> dest_host) const {
            if (size_ == 0) return;
            ROCCU_SAFE_CALL(cuMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
        }

        void to_host(std::span<Contained> dest_host, CUstream stream) const {
			if (size_ == 0) return;
            ROCCU_SAFE_CALL(cuMemcpyDtoHAsync(dest_host.data(), ptr_, min_size(dest_host.size()), stream));
		}

        auto to_host() const -> std::vector<Contained> {
			if (size_ == 0) return {};
			auto ret = std::vector<Contained>(size_);
			to_host(ret);
			return ret;
		}

        auto to_host(CUstream stream) const -> std::vector<Contained> {
            if (size_ == 0) return {};
            auto ret = std::vector<Contained>(size_);
            to_host(ret, stream);
            return ret;
        }

        void from_host(std::span<const Contained> src_host) {
			if (size_ == 0) return;
            ROCCU_SAFE_CALL(cuMemcpyHtoD(ptr_, src_host.data(), min_size(src_host.size())));
		}

        void from_host(std::span<const Contained> src_host, CUstream stream) {
            if (size_ == 0) return;
            ROCCU_SAFE_CALL(cuMemcpyHtoDAsync(ptr_, src_host.data(), min_size(src_host.size()), stream));
        }

        void free_async(CUstream stream) requires (is_owner) {
            if (ptr_) {
                ROCCU_SAFE_CALL(cuMemFreeAsync(ptr_, stream));
            }

            ptr_ = 0;
            size_ = 0;
        }

    private:

        constexpr auto min_size(std::size_t other_size) const noexcept {
            using namespace std;
            return min(other_size, size_) * sizeof(Contained);
        }

        CUdeviceptr ptr_ = 0;
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

        gpu_image_base(std::size_t width, std::size_t height) requires (is_owner) : width_(width), height_(height), pitch_(width) {
			ROCCU_SAFE_CALL(cuMemAlloc(&ptr_, size_bytes()));
		}

        gpu_image_base(std::size_t width, std::size_t height, CUstream stream) requires (is_owner) : width_(width), height_(height), pitch_(width) {
            ROCCU_SAFE_CALL(cuMemAllocAsync(&ptr_, size_bytes(), stream));
        }

        gpu_image_base(CUdeviceptr ptr, std::size_t width, std::size_t height) requires (!is_owner) : ptr_(ptr), width_(width), height_(height), pitch_(width) {}

        ~gpu_image_base() {
			if constexpr (!is_owner) return;
			if (ptr_) {
				cuMemFree(ptr_);
			}
		}

		gpu_image_base(const gpu_image_base&) requires (is_owner) = delete;
		gpu_image_base& operator=(const gpu_image_base&) requires (is_owner) = delete;

		gpu_image_base(const gpu_image_base&) requires (!is_owner) = default;
		gpu_image_base& operator=(const gpu_image_base&) requires (!is_owner) = default;

        gpu_image_base(gpu_image_base&& o) noexcept {
            std::swap(ptr_, o.ptr_);
            std::swap(width_, o.width_);
            std::swap(height_, o.height_);
            std::swap(pitch_, o.pitch_);
        }

        gpu_image_base& operator=(gpu_image_base&& o) noexcept {
			std::swap(ptr_, o.ptr_);
            std::swap(width_, o.width_);
            std::swap(height_, o.height_);
			std::swap(pitch_, o.pitch_);
			return *this;
		}

        explicit(false) operator gpu_image_base<PixelType, buffer_ownership::view>() requires (is_owner) {
            return { ptr_, width_, height_ };
        }

        constexpr auto ptr() const noexcept { return ptr_; }
        constexpr auto width() const noexcept { return width_; }
        constexpr auto height() const noexcept { return height_; }
        constexpr auto area() const noexcept { return width_ * height_; }
        constexpr auto pitch() const noexcept { return pitch_; }
        constexpr auto size_bytes() const noexcept { return area() * element_size; }

        constexpr bool valid() const noexcept { return ptr_ != 0; }
        constexpr explicit operator bool() const noexcept { return valid(); }

        void clear() requires (is_owner) {
            ROCCU_SAFE_CALL(cuMemsetD8(ptr_, 0, size_bytes()));
        }

        void clear(CUstream stream) requires (is_owner) {
            ROCCU_SAFE_CALL(cuMemsetD8Async(ptr_, 0, size_bytes(), stream));
        }

        auto to_host(std::span<PixelType> dest_host) const {
            auto copy_param = to_host_memcpy(dest_host);
            ROCCU_SAFE_CALL(cuMemcpy2D(&copy_param));
        }

        auto to_host(std::span<PixelType> dest_host, CUstream stream) const {
            auto copy_param = to_host_memcpy(dest_host);
			ROCCU_SAFE_CALL(cuMemcpy2DAsync(&copy_param, stream));
		}

        auto to_host() const -> std::vector<PixelType> {
            auto ret = std::vector<PixelType>(area());
            to_host(ret);
            return ret;
        }

        auto to_host(CUstream stream) const -> std::vector<PixelType>  {
			auto ret = std::vector<PixelType>(area());
			to_host(ret, stream);
			return ret;
		}

        auto to_host_flat() const -> std::vector<std::byte> {
            auto ret = std::vector<std::byte>{};
            auto size = size_bytes();
            ret.resize(size);
            ROCCU_SAFE_CALL(cuMemcpyDtoH(ret.data(), ptr_, size_bytes()));
            return ret;
        }

        void from_host(std::span<const PixelType> src_host)  {
            auto copy_param = from_host_memcpy(src_host);
			ROCCU_SAFE_CALL(cuMemcpy2D(&copy_param));
		}

        void from_host(std::span<const PixelType> src_host, CUstream stream) {
            auto copy_param = from_host_memcpy(src_host);
            ROCCU_SAFE_CALL(cuMemcpy2DAsync(&copy_param, stream));
        }

        void free_async(CUstream stream) requires (is_owner) {
			if (ptr_) {
				ROCCU_SAFE_CALL(cuMemFreeAsync(ptr_, stream));
			}

			ptr_ = 0;
            width_ = 0;
            height_ = 0;
			pitch_ = 0;
		}

        /*auto sub_image(uint2 offset, uint2 dims) -> gpu_image_base<PixelType, buffer_ownership::view> {
            auto new_ptr = ptr_ + element_size * (offset.y * pitch_ + offset.x);
			return { new_ptr, dims, pitch_ };
		}*/

    private:

        auto to_host_memcpy(std::span<PixelType> dest_host) const {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = ptr_;
            copy.srcPitch = pitch_ * element_size;
            copy.srcXInBytes = 0;
            copy.srcY = 0;

            copy.dstMemoryType = CU_MEMORYTYPE_HOST;
            copy.dstHost = dest_host.data();
            copy.dstPitch = width_ * element_size;
            copy.dstXInBytes = 0;
            copy.dstY = 0;

            copy.WidthInBytes = width_ * element_size;
            copy.Height = height_;

            return copy;
		}
        
        auto from_host_memcpy(std::span<const PixelType> src_host) const {
			CUDA_MEMCPY2D copy{};
			copy.srcMemoryType = CU_MEMORYTYPE_HOST;
			copy.srcHost = src_host.data();
			copy.srcPitch = width_ * element_size;
			copy.srcXInBytes = 0;
			copy.srcY = 0;

			copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			copy.dstDevice = ptr_;
			copy.dstPitch = pitch_ * element_size;
			copy.dstXInBytes = 0;
			copy.dstY = 0;

			copy.WidthInBytes = width_ * element_size;
			copy.Height = height_;

			return copy;
		}

        CUdeviceptr ptr_ = 0;
        std::size_t width_ = 0;
        std::size_t height_ = 0;
        std::size_t pitch_ = 0;
    };

    template<typename PixelType>
    using gpu_image = gpu_image_base<PixelType, buffer_ownership::owner>;

    template<typename PixelType>
    using gpu_image_view = gpu_image_base<PixelType, buffer_ownership::view>;
}