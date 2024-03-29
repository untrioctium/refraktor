#pragma once

#include <roccu_cpp_types.h>

namespace rfkt{

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

		[[nodiscard]] explicit(false) operator roccu::gpu_span<PixelType>() const noexcept {
			return roccu::gpu_span<PixelType>(buffer);
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
		roccu::gpu_buffer<PixelType> buffer;
	};

}
