#ifndef EZNVE_HPP
#define EZNVE_HPP

#include <cuda.h>
#include <vector_types.h>

#include <functional>
#include <span>
#include <array>
#include <memory>

#include <assert.h>

namespace eznve {

	enum class codec {
		h264,
		hevc
	};

	enum class frame_flag {
		none,
		idr
	};

	struct chunk {
		std::vector<char> data;
		uint32_t index;
		uint64_t timestamp;
		uint64_t duration;
	};

	class encoder {
	public:
		encoder(uint2 dims, uint2 fps, codec c, CUcontext ctx);
		~encoder();

		encoder(const encoder&) = delete;
		encoder& operator=(const encoder&) = delete;

		encoder(encoder&&) noexcept = default;
		encoder& operator=(encoder&&) = default;

		std::optional<chunk> submit_frame(frame_flag = frame_flag::none);
		std::optional<chunk> flush();

		CUdeviceptr buffer() const noexcept {
			return input_buffer;
		}

		auto buffer_size() const noexcept {
			return width() * height() * 4;
		}

		// width of the encoded video
		std::size_t width() const noexcept {
			return dims.x;
		}

		// height of the encoded video
		std::size_t height() const noexcept {
			return dims.y;
		}

		// returns the encoder fps as a double
		// this should only be used as a rough estimate
		// as fps_exact() returns the actual numerator and denominator
		double fps() const noexcept {
			return double(fps_.x)/fps_.y;
		}

		auto fps_exact() const noexcept {
			return fps_;
		}

		// returns the total bytes emitted from the encoder
		// since the beginning or the last flush
		auto total_bytes() const noexcept {
			return bytes_encoded;
		}

		// returns the total frames processed by the encoder
		// since the beginning or the last flush
		auto total_frames() const noexcept {
			return frames_encoded;
		}

		// returns the current time on the encoder in seconds
		double time() const noexcept {
			return total_frames() / fps();
		}

	private:

		using param_buffer_t = std::array<std::byte, 16384>;
		std::unique_ptr<param_buffer_t> pbuf = std::make_unique<param_buffer_t>();

		template<typename T>
		T* pbuf_as() noexcept {
			assert(sizeof(T) < pbuf->size());
			memset(pbuf->data(), 0, sizeof(T));
			return reinterpret_cast<T*>(pbuf->data());
		}

		CUdeviceptr input_buffer = 0;
		uint2 dims = {0,0};
		uint2 fps_ = {0,0};

		void* in_registration = nullptr;
		void* out_stream = nullptr;
		void* session = nullptr;

		std::size_t bytes_encoded = 0;
		std::uint32_t frames_encoded = 0;
	};

}

#endif