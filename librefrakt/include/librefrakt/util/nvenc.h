#pragma once

#include <memory>
#include <optional>
#include <fmt/format.h>

#include <librefrakt/util/cuda.h>
#include <librefrakt/cuda_buffer.h>


namespace rfkt::nvenc {

	using session_handle_t = void*;

	enum class codec {
		h264,
		hevc
	};

	enum class buffer_format {
		undefined,
		nv12,
		yv12,
		iyuv,
		yuv444,
		yuv420_10bit,
		yuv444_10bit,
		argb,
		argb10,
		ayuv,
		abgr,
		abgr10,
		u8
	};

	class session {
	public:
		static std::unique_ptr<session> make();

		std::string last_error();

		inline void print_last_error() {
			fmt::print("{}", last_error());
		}

		std::shared_ptr<cuda_buffer<uchar4>> initialize(std::pair<uint32_t, uint32_t> dims, std::pair<uint32_t, uint32_t> fps);
		std::optional<std::vector<unsigned char>> submit_frame(bool idr, bool done = false);
		~session();

	private:

		std::pair<uint32_t, uint32_t> dims_;
		void* in_reg;
		std::shared_ptr<cuda_buffer<uchar4>> input_buffer;
		void* out_stream;
		void* sesh;
	};


}