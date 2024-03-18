#pragma once

#include <librefrakt/util/cuda.h>
#include <span>

#include <spdlog/spdlog.h>
#include <librefrakt/util.h>

namespace rfkt {

	template<class Contained = char>
	class gpu_buffer {
	public:
		gpu_buffer() noexcept = default;

		explicit gpu_buffer(std::size_t size) : 
			size_(size)
		{
			ruMemAlloc(&ptr_, size_ * sizeof(Contained));
		}

		gpu_buffer(std::size_t size, RUstream stream) :
			size_(size)
		{
			CUDA_SAFE_CALL(ruMemAllocAsync(&ptr_, size_ * sizeof(Contained), stream));
		}

		~gpu_buffer() noexcept {
			if (ptr_) {
				CUDA_SAFE_CALL(ruMemFree(ptr_));
			}	
		}

		gpu_buffer(const gpu_buffer&) = delete;
		gpu_buffer& operator=(const gpu_buffer&) = delete;

		constexpr auto ptr() const noexcept { return ptr_; }
		constexpr auto size() const noexcept { return size_; }
		constexpr auto size_bytes() const noexcept { return size_ * sizeof(Contained); }

		constexpr explicit operator bool() const { return ptr_ != 0; }
		constexpr bool valid() const { return ptr_ != 0; }

		constexpr gpu_buffer& operator=(gpu_buffer&& o) noexcept {
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
			return *this;
		}

		constexpr gpu_buffer(gpu_buffer&& o) noexcept {
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
		}

		void clear() {
			CUDA_SAFE_CALL(ruMemsetD8(ptr_, 0, size_bytes()));
		}

		void clear(RUstream stream) {
			CUDA_SAFE_CALL(ruMemsetD8Async(ptr_, 0, size_bytes(), stream));
		}

		void to_host(std::span<Contained> dest_host) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(ruMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
		}

		void to_host(std::span<Contained> dest_host, RUstream stream) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(ruMemcpyDtoHAsync(dest_host.data(), ptr_, min_size(dest_host.size()), stream));
		}

		void from_host(std::span<const Contained> src_host) {
			CUDA_SAFE_CALL(ruMemcpyHtoD(ptr_, src_host.data(), min_size(src_host.size())));
		}

		void from_host(std::span<const Contained> src_host, RUstream stream) {
			CUDA_SAFE_CALL(ruMemcpyHtoDAsync(ptr_, src_host.data(), min_size(src_host.size()), stream));
		}

		void free_async(RUstream stream) {
			if (ptr_) {
				CUDA_SAFE_CALL(ruMemFreeAsync(ptr_, stream));
			}

			ptr_ = 0;
			size_ = 0;
		}

	private:

		constexpr auto min_size(std::size_t other_size) const noexcept {
			using namespace std;
			return min(other_size, size_) * sizeof(Contained);
		}

		constexpr gpu_buffer(RUdeviceptr ptr, std::size_t size) noexcept :
			size_(size),
			ptr_(ptr) {}

		RUdeviceptr ptr_ = 0;
		std::size_t size_ = 0;
	};

	template<typename Contained = char>
	class gpu_span {
	public:
		constexpr gpu_span() noexcept = default;

		constexpr gpu_span(RUdeviceptr ptr, std::size_t size) noexcept :
			ptr_(ptr),
			size_(size) {}

		explicit(false) gpu_span(const gpu_buffer<Contained>& buf) noexcept :
			ptr_(buf.ptr()),
			size_(buf.size()) {}

		~gpu_span() noexcept = default;

		constexpr gpu_span(const gpu_span&) noexcept = default;
		constexpr gpu_span& operator=(const gpu_span&) noexcept = default;

		constexpr gpu_span& operator=(gpu_span&& o) noexcept = default;
		constexpr gpu_span(gpu_span&& o) noexcept = default;

		[[nodiscard]] constexpr auto ptr() const noexcept { return ptr_; }
		[[nodiscard]] constexpr auto size() const noexcept { return size_; }
		[[nodiscard]] constexpr auto size_bytes() const noexcept { return size_ * sizeof(Contained); }
		[[nodiscard]] constexpr bool valid() const noexcept { return ptr_ != 0 && size_ != 0; }
		[[nodiscard]] constexpr explicit operator bool() const noexcept { return valid(); }

		void clear() {
			CUDA_SAFE_CALL(ruMemsetD8(ptr_, 0, size_bytes()));
		}

		void clear(RUstream stream) {
			CUDA_SAFE_CALL(ruMemsetD8Async(ptr_, 0, size_bytes(), stream));
		}

		void to_host(std::span<Contained> dest_host) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(ruMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
		}

		void to_host(std::span<Contained> dest_host, RUstream stream) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(ruMemcpyDtoHAsync(dest_host.data(), ptr_, min_size(dest_host.size()), stream));
		}

		void from_host(std::span<const Contained> src_host) {
			CUDA_SAFE_CALL(ruMemcpyHtoD(ptr_, src_host.data(), min_size(src_host.size())));
		}

		void from_host(std::span<const Contained> src_host, RUstream stream) {
			CUDA_SAFE_CALL(ruMemcpyHtoDAsync(ptr_, src_host.data(), min_size(src_host.size()), stream));
		}

	private:

		constexpr auto min_size(std::size_t other_size) const noexcept {
			using namespace std;
			return min(other_size, size_) * sizeof(Contained);
		}

		RUdeviceptr ptr_ = 0;
		std::size_t size_ = 0;
	};

	template<typename PixelType>
	class gpu_image {
	public:
		gpu_image() = default;
		gpu_image(unsigned int w, unsigned int h) :
			dims_(w, h),
			buffer(area()) {}

		gpu_image(unsigned int w, unsigned int h, RUstream stream) :
			dims_(w, h),
			buffer(area(), stream) {}

		~gpu_image() = default;

		gpu_image(const gpu_image&) = delete;
		gpu_image& operator=(const gpu_image&) = delete;

		gpu_image(gpu_image&& o) noexcept {
			std::swap(dims_, o.dims_);
			std::swap(buffer, o.buffer);
		}
		gpu_image& operator=(gpu_image&& o) noexcept {
			std::swap(dims_, o.dims_);
			std::swap(buffer, o.buffer);
			return *this;
		}


		std::size_t area() const noexcept { return static_cast<std::size_t>(dims_.x) * dims_.y; }
		std::size_t size_bytes() const noexcept { return buffer.size_bytes(); }

		unsigned int width() const noexcept { return dims_.x; }
		unsigned int height() const noexcept { return dims_.y; }
		uint2 dims() const noexcept { return dims_; }

		[[nodiscard]] auto ptr() noexcept { return buffer.ptr(); }

		[[nodiscard]] auto ptr() const noexcept { return buffer.ptr(); }
		[[nodiscard]] bool valid() const noexcept { return buffer.valid(); }
		[[nodiscard]] explicit operator bool() const noexcept { return valid(); }

		[[nodiscard]] explicit(false) operator gpu_span<PixelType>() const noexcept {
			return gpu_span<PixelType>(buffer);
		}

		void clear() { buffer.clear(); }
		void clear(RUstream stream) { buffer.clear(stream); }

		void to_host(std::span<PixelType> dest_host) const { buffer.to_host(dest_host); }
		void to_host(std::span<PixelType> dest_host, RUstream stream) const { buffer.to_host(dest_host, stream); }

		void from_host(std::span<const PixelType> src_host) { buffer.from_host(src_host); }
		void from_host(std::span<const PixelType> src_host, RUstream stream) { buffer.from_host(src_host, stream); }

		void free_async(RUstream stream) { buffer.free_async(stream); }

	private:
		uint2 dims_ = { 0, 0 };
		gpu_buffer<PixelType> buffer;
	};

}
