#pragma once

#include <future>
#include <queue>
#include <librefrakt/util/cuda.h>

using nvjpegHandle_t = struct nvJpegHandle*;
using nvjpegEncoderState_t = struct nvjpegEncoderState*;
using nvjpegEncoderParams_t = struct nvJpegEncoderParams_st*;

namespace rfkt::nvjpeg {
	
	void init();

	class encoder {
	public:

		auto encode_image(RUdeviceptr image, int width, int height, int quality, RUstream stream) -> std::future<std::move_only_function<std::vector<std::byte>()>>;
		explicit encoder(RUstream stream);
		~encoder();

	private:
		auto wrap_state(nvjpegEncoderState_t state)->std::unique_ptr<nvjpegEncoderState, std::function<void(nvjpegEncoderState_t)>>;
		auto make_state(RUstream stream)->nvjpegEncoderState_t;

		auto make_params(unsigned char quality, RUstream stream)->nvjpegEncoderParams_t;

		std::mutex states_mutex;
		std::queue<nvjpegEncoderState_t> available_states;
		std::unordered_map<unsigned char, nvjpegEncoderParams_t> params_map;

		int max_states = 1;

		nvjpegHandle_t nv_handle;
		//nvjpegDevAllocator_t dev_alloc;
		//nvjpegPinnedAllocator_t p_alloc;
	};
}