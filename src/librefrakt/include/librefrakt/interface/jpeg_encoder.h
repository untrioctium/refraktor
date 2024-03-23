#pragma once

#include <future>
#include <set>
#include <librefrakt/util/cuda.h>
#include <librefrakt/gpu_buffer.h>
#include <librefrakt/factory.h>

namespace rfkt {

	struct jpeg_encoder : factory<jpeg_encoder, roccu::gpu_stream&> {
		struct meta_type {
			std::size_t priority;
			std::set<roccu_api> supported_apis;
		};

		explicit jpeg_encoder(key) {}
		virtual ~jpeg_encoder() = default;

		using encode_thunk = std::move_only_function<std::vector<std::byte>()>;
		virtual auto encode_image(const gpu_image<uchar3>& image, int quality, roccu::gpu_stream& stream) -> std::future<encode_thunk> = 0;
	};

}