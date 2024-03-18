#pragma once

#include <roccu.h>
#include <glad/glad.h>
#include <cstdint>
#include <atomic>
#include <variant>
#include <any>

#include <future>

#include <librefrakt/traits/noncopyable.h>
#include <librefrakt/gpu_buffer.h>

namespace rfkt::gl {

	enum class texture_format : unsigned char {
		rgba8,
		rgb8,
		rgba16f,
		rgb16f,
		rgba32f,
		rgb32f,
	};

	enum class sampling_mode : GLenum {
		nearest = GL_NEAREST,
		linear = GL_LINEAR,
	};

	template<texture_format Format>
	struct texture_format_traits;

#define RFKT_TEX_FORMAT_TRAITS(Format, Channels, Type, Internal_Type) \
	template<> \
	struct texture_format_traits<texture_format::Format> { \
		static constexpr std::size_t channels = Channels; \
		static constexpr GLenum internal_type = Internal_Type; \
		using pixel_type = Type; \
	}

	RFKT_TEX_FORMAT_TRAITS(rgba8, 4, uchar4, GL_RGBA8);
	RFKT_TEX_FORMAT_TRAITS(rgb8, 3, uchar3, GL_RGB8);
	RFKT_TEX_FORMAT_TRAITS(rgba16f, 4, short4, GL_RGBA16F);
	RFKT_TEX_FORMAT_TRAITS(rgb16f, 3, half3, GL_RGB16F);
	RFKT_TEX_FORMAT_TRAITS(rgba32f, 4, float4, GL_RGBA32F);
	RFKT_TEX_FORMAT_TRAITS(rgb32f, 3, float3, GL_RGB32F);

#undef RFKT_TEX_FORMAT_TRAITS

	namespace detail {
		std::pair<GLuint, RUgraphicsResource> allocate_texture(std::size_t w, std::size_t h, GLenum internal_format, sampling_mode s_mode);
		void deallocate_texture(GLuint id, RUgraphicsResource res);

		std::pair<RU_MEMCPY2D, RUarray> create_mapping(RUgraphicsResource cuda_res, std::size_t w, std::size_t h, std::size_t pixel_size);
		void destroy_mapping(RUgraphicsResource cuda_res, std::any&& parent);
	}

	template<texture_format Format>
	class texture : public std::enable_shared_from_this<texture<Format>>, public rfkt::traits::noncopyable {
	public:

		using handle = std::shared_ptr<texture>;
		static constexpr texture_format format = Format;
		using traits = texture_format_traits<Format>;

		static handle create(std::size_t w, std::size_t h, sampling_mode s_mode) {
			return handle{ new texture{w, h, s_mode} };
		}

		~texture() {
			detail::deallocate_texture(tex_id, cuda_res);
		}

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

			cuda_map() = delete;

			~cuda_map() {
				if(tex)
					detail::destroy_mapping(tex->cuda_res, tex);
			}

			cuda_map(cuda_map&& o) noexcept {
				this->swap(o);
			}

			cuda_map& operator=(cuda_map&& o) noexcept {
				this->swap(o);
				return *this;
			}

			void copy_from(gpu_span<typename texture::traits::pixel_type> buffer) {
				if(buffer.size_bytes() < copy_params.WidthInBytes * copy_params.Height)
					throw std::runtime_error("cuda_map::copy_from: buffer too small");

				copy_params.srcDevice = buffer.ptr();
				CUDA_SAFE_CALL(ruMemcpy2D(&copy_params));
			}

			void copy_from(gpu_span<typename texture::traits::pixel_type> buffer, rfkt::gpu_stream& stream) {
				if (buffer.size_bytes() < copy_params.WidthInBytes * copy_params.Height)
					throw std::runtime_error("cuda_map::copy_from: buffer too small");

				copy_params.srcDevice = buffer.ptr();
				CUDA_SAFE_CALL(ruMemcpy2DAsync(&copy_params, stream));
			}
		private:

			cuda_map(texture::handle&& parent) : tex(std::move(parent)) {
				auto [p, a] = detail::create_mapping(tex->cuda_res, tex->width(), tex->height(), sizeof(typename texture::traits::pixel_type));
				copy_params = p;
				arr = a;
			}

			friend class texture;

			RU_MEMCPY2D copy_params;
			RUarray arr = nullptr;

			texture::handle tex;

			void swap(cuda_map& o) {
				std::swap(copy_params, o.copy_params);
				std::swap(arr, o.arr);
				std::swap(tex, o.tex);
			}
		};

		GLuint id() const { return tex_id; }
		std::size_t area() const { return width_ * height_; }
		std::size_t width() const { return width_; }
		std::size_t height() const { return height_; }

		auto map_to_cuda() { return cuda_map(this->shared_from_this()); }
	private:

		texture() = default;
		texture(std::size_t w, std::size_t h, sampling_mode s_mode): width_(w), height_(h) {
			auto [id, res] = detail::allocate_texture(w, h, traits::internal_type, s_mode);
			tex_id = id;
			cuda_res = res;
		}

		void swap(texture& o) {
			std::swap(tex_id, o.tex_id);
			std::swap(cuda_res, o.cuda_res);
			std::swap(width_, o.width_);
			std::swap(height_, o.height_);
		}

		friend class cuda_map;

		GLuint tex_id = 0;
		RUgraphicsResource cuda_res = nullptr;

		std::size_t width_;
		std::size_t height_;
	};
}