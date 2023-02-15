#pragma once

#include <cuda.h>
#include <glad/glad.h>
#include <cstdint>

struct GLFWwindow;
struct GLFWmonitor;

namespace rfkt::gl {

	bool init(int width, int height);
	void begin_frame();
	void end_frame(bool render);
	bool close_requested();

	class texture {
	public:
		texture(std::size_t w, std::size_t h);
		~texture();

		texture(const texture&) = delete;
		texture(texture&& o) {
			std::swap(tex_id, o.tex_id);
			std::swap(cuda_res, o.cuda_res);
			std::swap(width_, o.width_);
			std::swap(height_, o.height_);
		}

		class
			[[nodiscard("texture::cuda_map should be held")]]
		cuda_map {
		public:
			cuda_map(texture& tex);
			~cuda_map();

			cuda_map(const cuda_map&) = delete;
			cuda_map& operator=(const cuda_map&) = delete;
			cuda_map(cuda_map&&) = default;
			cuda_map& operator=(cuda_map&&) = default;

			void copy_from(CUdeviceptr ptr);
		private:
			CUDA_MEMCPY2D copy_params;
			CUgraphicsResource cuda_res = nullptr;
			CUarray arr = nullptr;
		};

		GLuint id() const { return tex_id; }
		std::size_t area() const { return width_ * height_; }
		std::size_t width() const { return width_; }
		std::size_t height() const { return height_; }

		auto map_to_cuda() { return cuda_map(*this); }
	private:
		friend class cuda_map;

		GLuint tex_id = 0;
		CUgraphicsResource cuda_res = nullptr;

		std::size_t width_;
		std::size_t height_;
	};
}