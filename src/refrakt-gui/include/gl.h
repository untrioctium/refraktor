#pragma once

#include <cuda.h>
#include <glad/glad.h>
#include <cstdint>
#include <atomic>
#include <variant>

#include <future>

#include <librefrakt/traits/noncopyable.h>
#include <librefrakt/cuda_buffer.h>

namespace rfkt::gl {

	int2 get_window_size();
	void make_current();

	bool init(int width, int height);
	void begin_frame();
	void end_frame(bool render);
	bool close_requested();

	void event_loop(std::stop_token);

	std::string show_open_dialog(std::string_view filter = {});
	void set_clipboard(std::string contents);
	auto get_clipboard() -> std::future<std::string>;
	void set_mouse_position(double x, double y);
	void set_cursor_enabled(bool enabled);
	bool cursor_enabled();
	void set_window_title(std::string_view title);

	void set_target_fps(unsigned int fps = 0);

	class texture : public rfkt::traits::noncopyable {
	public:
		texture(std::size_t w, std::size_t h);
		~texture();

		texture(texture&& o) noexcept {
			this->swap(o);
		}

		texture& operator=(texture&& o) noexcept {
			this->swap(o);
			return *this;
		}

		class
			[[nodiscard("texture::cuda_map should be held")]]
		cuda_map : public rfkt::traits::noncopyable {
		public:
			cuda_map(texture& tex);
			~cuda_map();

			cuda_map(cuda_map&& o) noexcept {
				this->swap(o);
			}

			cuda_map& operator=(cuda_map&& o) noexcept {
				this->swap(o);
				return *this;
			}

			template<typename T>
			void copy_from(const cuda_buffer<T>& buffer) {
				if(buffer.size_bytes() < copy_params.WidthInBytes * copy_params.Height)
					throw std::runtime_error("cuda_map::copy_from: buffer too small");

				copy_params.srcDevice = buffer.ptr();
				CUDA_SAFE_CALL(cuMemcpy2D(&copy_params));
			}

			template<typename T>
			void copy_from(const cuda_buffer<T>& buffer, rfkt::cuda_stream& stream) {
				if (buffer.size_bytes() < copy_params.WidthInBytes * copy_params.Height)
					throw std::runtime_error("cuda_map::copy_from: buffer too small");

				copy_params.srcDevice = buffer.ptr();
				CUDA_SAFE_CALL(cuMemcpy2DAsync(&copy_params, stream));
			}
		private:
			CUDA_MEMCPY2D copy_params;
			CUgraphicsResource cuda_res = nullptr;
			CUarray arr = nullptr;

			void swap(cuda_map& o) {
				std::swap(copy_params, o.copy_params);
				std::swap(cuda_res, o.cuda_res);
				std::swap(arr, o.arr);
			}
		};

		GLuint id() const { return tex_id; }
		std::size_t area() const { return width_ * height_; }
		std::size_t width() const { return width_; }
		std::size_t height() const { return height_; }

		auto map_to_cuda() { return cuda_map(*this); }
	private:

		void swap(texture& o) {
			std::swap(tex_id, o.tex_id);
			std::swap(cuda_res, o.cuda_res);
			std::swap(width_, o.width_);
			std::swap(height_, o.height_);
		}

		friend class cuda_map;

		GLuint tex_id = 0;
		CUgraphicsResource cuda_res = nullptr;

		std::size_t width_;
		std::size_t height_;
	};

	static_assert(!std::is_copy_constructible_v<texture>);
	static_assert(!std::is_copy_assignable_v<texture>);
	static_assert(std::is_move_constructible_v<texture>);
	static_assert(std::is_move_assignable_v<texture>);

	static_assert(!std::is_copy_constructible_v<texture::cuda_map>);
	static_assert(!std::is_copy_assignable_v<texture::cuda_map>);
	static_assert(std::is_move_constructible_v<texture::cuda_map>);
	static_assert(std::is_move_assignable_v<texture::cuda_map>);
}