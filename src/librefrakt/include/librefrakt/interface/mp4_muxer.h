#pragma once

#include <librefrakt/factory.h>
#include <span>

namespace rfkt {

	struct mp4_muxer : factory<mp4_muxer, std::string, int> {
		explicit mp4_muxer(key) {}
		virtual ~mp4_muxer() = default;

		virtual void write_chunk(std::span<const std::byte> chunk) = 0;
		virtual void finish() = 0;
	};

}