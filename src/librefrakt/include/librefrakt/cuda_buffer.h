#pragma once

#include <librefrakt/util/cuda.h>
#include <span>

namespace rfkt {

	template<class Contained = char>
	class cuda_buffer {
	public:
		cuda_buffer() noexcept = default;

		explicit cuda_buffer(std::size_t size) : 
			size_(size)
		{
			cuMemAlloc(&ptr_, size_ * sizeof(Contained));
		}

		cuda_buffer(std::size_t size, CUstream stream) :
			size_(size)
		{
			CUDA_SAFE_CALL(cuMemAllocAsync(&ptr_, size_ * sizeof(Contained), stream));
		}

		~cuda_buffer() noexcept {
			if (ptr_) {
				CUDA_SAFE_CALL(cuMemFree(ptr_));
			}	
		}

		cuda_buffer(const cuda_buffer&) = delete;
		cuda_buffer& operator=(const cuda_buffer&) = delete;

		auto ptr() const noexcept { return ptr_; }
		auto size() const noexcept { return size_; }
		auto size_bytes() const noexcept { return size_ * sizeof(Contained); }

		explicit operator bool() const { return ptr_ != 0; }
		bool valid() const { return ptr_ != 0; }

		cuda_buffer& operator=(cuda_buffer&& o) noexcept {
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
			return *this;
		}

		cuda_buffer(cuda_buffer&& o) noexcept {
			std::swap(ptr_, o.ptr_);
			std::swap(size_, o.size_);
		}

		void clear() {
			CUDA_SAFE_CALL(cuMemsetD8(ptr_, 0, size_bytes()));
		}

		void clear(CUstream stream) {
			CUDA_SAFE_CALL(cuMemsetD8Async(ptr_, 0, size_bytes(), stream));
		}

		void to_host(std::span<Contained> dest_host) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(cuMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
		}

		void to_host(std::span<Contained> dest_host, CUstream stream) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(cuMemcpyDtoHAsync(dest_host.data(), ptr_, min_size(dest_host.size()), stream));
		}

		void from_host(std::span<const Contained> src_host) {
			CUDA_SAFE_CALL(cuMemcpyHtoD(ptr_, src_host.data(), min_size(src_host.size())));
		}

		void from_host(std::span<const Contained> src_host, CUstream stream) {
			CUDA_SAFE_CALL(cuMemcpyHtoDAsync(ptr_, src_host.data(), min_size(src_host.size()), stream));
		}

		void free_async(CUstream stream) {
			if (ptr_) {
				CUDA_SAFE_CALL(cuMemFreeAsync(ptr_, stream));
			}

			ptr_ = 0;
			size_ = 0;
		}

	private:

		auto min_size(std::size_t other_size) const noexcept {
			using namespace std;
			return min(other_size, size_) * sizeof(Contained);
		}

		cuda_buffer(CUdeviceptr ptr, std::size_t size) noexcept :
			size_(size),
			ptr_(ptr) {}

		CUdeviceptr ptr_ = 0;
		std::size_t size_ = 0;
	};

	template<typename Contained = char>
	class cuda_span {
	public:
		cuda_span() noexcept = default;

		cuda_span(CUdeviceptr ptr, std::size_t size_bytes) noexcept :
			ptr_(ptr),
			size_(size_bytes/sizeof(Contained)) {}

		explicit(false) cuda_span(const cuda_buffer<Contained>& buf) noexcept :
			ptr_(buf.ptr()),
			size_(buf.size()) {}

		~cuda_span() noexcept = default;

		cuda_span(const cuda_span&) noexcept = default;
		cuda_span& operator=(const cuda_span&) noexcept = default;

		cuda_span& operator=(cuda_span&& o) noexcept = default;
		cuda_span(cuda_span&& o) noexcept = default;

		[[nodiscard]] auto ptr() const noexcept { return ptr_; }
		[[nodiscard]] auto size() const noexcept { return size_; }
		[[nodiscard]] auto size_bytes() const noexcept { return size_ * sizeof(Contained); }
		[[nodiscard]] bool valid() const noexcept { return ptr_ != 0 && size_ != 0; }
		[[nodiscard]] explicit operator bool() const noexcept { return valid(); }

		void clear() {
			CUDA_SAFE_CALL(cuMemsetD8(ptr_, 0, size_bytes()));
		}

		void clear(CUstream stream) {
			CUDA_SAFE_CALL(cuMemsetD8Async(ptr_, 0, size_bytes(), stream));
		}

		void to_host(std::span<Contained> dest_host) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(cuMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
		}

		void to_host(std::span<Contained> dest_host, CUstream stream) const {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(cuMemcpyDtoHAsync(dest_host.data(), ptr_, min_size(dest_host.size()), stream));
		}

		void from_host(std::span<Contained> src_host) {
			CUDA_SAFE_CALL(cuMemcpyHtoD(ptr_, src_host.data(), min_size(src_host.size())));
		}

		void from_host(std::span<Contained> src_host, CUstream stream) {
			CUDA_SAFE_CALL(cuMemcpyHtoDAsync(ptr_, src_host.data(), min_size(src_host.size()), stream));
		}

	private:

		auto min_size(std::size_t other_size) const noexcept {
			using namespace std;
			return min(other_size, size_) * sizeof(Contained);
		}

		CUdeviceptr ptr_ = 0;
		std::size_t size_ = 0;
	};

	template<typename PixelType>
	class cuda_image {
	public:
		cuda_image() = default;
		cuda_image(unsigned int w, unsigned int h) :
			dims_(w, h),
			buffer(w * h) {}

		cuda_image(unsigned int w, unsigned int h, CUstream stream) :
			dims_(w, h),
			buffer(w * h, stream) {}

		~cuda_image() = default;

		cuda_image(const cuda_image&) = delete;
		cuda_image& operator=(const cuda_image&) = delete;

		cuda_image(cuda_image&&) = default;
		cuda_image& operator=(cuda_image&&) = default;

		std::size_t area() const noexcept { return static_cast<std::size_t>(dims_.x) * dims_.y; }
		std::size_t size_bytes() const noexcept { return buffer.size_bytes(); }

		unsigned int width() const noexcept { return dims_.x; }
		unsigned int height() const noexcept { return dims_.y; }
		uint2 dims() const noexcept { return dims_; }

		[[nodiscard]] auto ptr() noexcept { return buffer.ptr(); }

		[[nodiscard]] auto ptr() const noexcept { return buffer.ptr(); }
		[[nodiscard]] bool valid() const noexcept { return buffer.valid(); }
		[[nodiscard]] explicit operator bool() const noexcept { return valid(); }

		[[nodiscard]] explicit(false) operator cuda_span<PixelType>() const noexcept {
			return cuda_span<PixelType>(buffer);
		}

		void clear() { buffer.clear(); }
		void clear(CUstream stream) { buffer.clear(stream); }

		void to_host(std::span<PixelType> dest_host) const { buffer.to_host(dest_host); }
		void to_host(std::span<PixelType> dest_host, CUstream stream) const { buffer.to_host(dest_host, stream); }

		void from_host(std::span<const PixelType> src_host) { buffer.from_host(src_host); }
		void from_host(std::span<const PixelType> src_host, CUstream stream) { buffer.from_host(src_host, stream); }

		void free_async(CUstream stream) { buffer.free_async(stream); }

	private:
		uint2 dims_ = { 0, 0 };
		cuda_buffer<PixelType> buffer;
	};

}
