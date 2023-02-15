#pragma once

#include <librefrakt/util/cuda.h>
#include <span>

namespace rfkt {

	template<class Contained = char>
	class cuda_buffer {
	public:
		cuda_buffer() noexcept = default;

		cuda_buffer(std::size_t size) : 
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

		explicit operator bool() { return ptr_ != 0; }

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
	class cuda_view {
	public:
		cuda_view() noexcept = default;

		cuda_view(CUdeviceptr ptr, std::size_t size_bytes) noexcept :
			ptr_(ptr),
			size_(size_bytes/sizeof(Contained)) {}

		cuda_view(const cuda_buffer<Contained>& buf) noexcept :
			ptr_(buf.ptr()),
			size_(buf.size()) {}

		~cuda_view() noexcept = default;

		cuda_view(const cuda_view&) noexcept = default;
		cuda_view& operator=(const cuda_view&) noexcept = default;

		cuda_view& operator=(cuda_view&& o) noexcept = default;
		cuda_view(cuda_view&& o) noexcept = default;

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

		void to_host(std::span<Contained> dest_host) {
			if (size_ == 0) return;
			CUDA_SAFE_CALL(cuMemcpyDtoH(dest_host.data(), ptr_, min_size(dest_host.size())));
		}

		void to_host(std::span<Contained> dest_host, CUstream stream) {
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

}
