#ifndef EZNVE_HPP
#define EZNVE_HPP

#include <roccu.hpp>

#include <functional>
#include <span>
#include <fmt/format.h>
#include <array>
#include <memory>
#include <queue>
#include <cstring>

#include <assert.h>

namespace eznve {

	struct uint2 {
		unsigned int x;
		unsigned int y;
	};

	enum class codec {
		h264,
		hevc,
		av1
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

	struct config {

		uint2 dims = {0,0};
		uint2 fps = {0,0};

		codec codec = codec::h264;

		enum class quality_preset {
			fastest,
			fast,
			balanced,
			quality,
			high_quality
		};

		enum class rate_control {
			cqp,
			vbr,
			cbr
		};

		enum class tuning {
			high_quality,
			low_latency,
			ultra_low_latency
		};

		quality_preset preset = quality_preset::high_quality;
		rate_control rc = rate_control::vbr;
		tuning tune = tuning::high_quality;

		uint32_t bitrate_kbps = 8000;
		uint32_t cqp = 23;
		uint32_t gop_length = 0;

		bool enable_bframes = true;
		bool enable_lookahead = true;
		uint32_t lookahead_depth = 20;

		static config default_config(uint2 dims, uint2 fps, enum codec c) {
			return config{
				.dims = dims,
				.fps = fps,
				.codec = c
			};
		}

		static config for_offline_rendering(uint2 dims, uint2 fps, enum codec c) {
			return config{
				.dims = dims,
				.fps = fps,
				.codec = c,
				.preset = quality_preset::high_quality,
				.rc = rate_control::cqp,
				.tune = tuning::high_quality,
				.cqp = 18,
				.enable_bframes = true,
				.enable_lookahead = true
			};
		}

		static config for_streaming(uint2 dims, uint2 fps, enum codec c) {
			return config{
				.dims = dims,
				.fps = fps,
				.codec = c,
				.preset = quality_preset::high_quality,
				.rc = rate_control::cbr,
				.tune = tuning::low_latency,
				.bitrate_kbps = 25000,
				.enable_bframes = false,
				.enable_lookahead = false,
			};
		}

	};

	class encoder {
	public:
		encoder(config cfg, CUcontext ctx, std::function<void(std::string_view)> logger = [](std::string_view) {});
		~encoder();

		encoder(const encoder&) = delete;
		encoder& operator=(const encoder&) = delete;

		encoder(encoder&& o) noexcept {
			*this = std::move(o);
		}

		encoder& operator=(encoder&& o) noexcept {
			std::swap(buffers, o.buffers);
			std::swap(free_buffers, o.free_buffers);
			std::swap(used_buffers, o.used_buffers);
			std::swap(dims, o.dims);
			std::swap(fps_, o.fps_);
			std::swap(bytes_encoded, o.bytes_encoded);
			std::swap(frames_encoded, o.frames_encoded);
			std::swap(session, o.session);
			std::swap(logger, o.logger);
			std::swap(max_in_flight, o.max_in_flight);
			std::swap(pbuf, o.pbuf);
			return *this;
		}

		std::vector<chunk> submit_frame(frame_flag = frame_flag::none);
		std::vector<chunk> flush();

		CUdeviceptr buffer() const noexcept {
			logger(fmt::format("giving buffer {}", free_buffers.front()));
			return buffers[free_buffers.front()].ptr;
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

		void push_buffer();

		struct buffer_t {
			CUdeviceptr ptr;
			void* registration;
			void* mapped;
			void* output_stream;

			void map(void* session);
			void unmap(void* session);

			chunk lock(void* session);
			void unlock(void* session);
		};


		std::vector<buffer_t> buffers;
		std::queue<std::size_t> free_buffers;
		std::queue<std::size_t> used_buffers;
		std::function<void(std::string_view)> logger;

		std::size_t max_in_flight = 0;

		using param_buffer_t = std::array<std::byte, 16384>;
		std::unique_ptr<param_buffer_t> pbuf = std::make_unique<param_buffer_t>();

		template<typename T>
		T* pbuf_as() noexcept {
			assert(sizeof(T) < pbuf->size());
			std::memset(pbuf->data(), 0, sizeof(T));
			return reinterpret_cast<T*>(pbuf->data());
		}

		uint2 dims = {0,0};
		uint2 fps_ = {0,0};

		void* session = nullptr;

		std::size_t bytes_encoded = 0;
		std::uint32_t frames_encoded = 0;
	};

}

#endif