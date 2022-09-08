#pragma once

#include <future>
#include <nvjpeg.h>
#include <queue>
#include <librefrakt/util/cuda.h>

namespace rfkt::nvjpeg {
	class encoder {
	public:

		auto encode_image(CUdeviceptr image, int width, int height, int quality, CUstream stream) -> std::future<std::vector<std::byte>>;
		encoder(CUstream stream);
		~encoder();

	private:

		auto wrap_state(nvjpegEncoderState_t state)->std::unique_ptr<nvjpegEncoderState, std::function<void(nvjpegEncoderState_t)>>;
		auto make_state(CUstream stream)->nvjpegEncoderState_t;

		auto make_params(unsigned char quality, CUstream stream)->nvjpegEncoderParams_t;

		std::mutex states_mutex;
		std::queue<nvjpegEncoderState_t> available_states;
		std::unordered_map<unsigned char, nvjpegEncoderParams_t> params_map;

		int max_states = 8;

		nvjpegHandle_t nv_handle;
	};
}