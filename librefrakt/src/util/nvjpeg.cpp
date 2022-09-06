#include <spdlog/spdlog.h>

#include <librefrakt/util/nvjpeg.h>


#define NVJPEG_SAFE_CALL(x)                                         \
  do {                                                            \
    nvjpegStatus_t result = x;                                          \
    if (result != NVJPEG_STATUS_SUCCESS) {                                 \
      SPDLOG_ERROR("`{}` failed with result: {}", #x, get_nvjpeg_error_string(result));       \
      exit(1);                                                    \
    }                                                             \
  } while(0)

const char* get_nvjpeg_error_string(nvjpegStatus_t status) {
	switch (status) {
	case NVJPEG_STATUS_SUCCESS: return "NVJPEG_STATUS_SUCCESS";
	case NVJPEG_STATUS_NOT_INITIALIZED: return "NVJPEG_STATUS_NOT_INITIALIZED";
	case NVJPEG_STATUS_INVALID_PARAMETER: return "NVJPEG_STATUS_INVALID_PARAMETER";
	case NVJPEG_STATUS_BAD_JPEG: return "NVJPEG_STATUS_BAD_JPEG";
	case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
	case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
	case NVJPEG_STATUS_EXECUTION_FAILED: return "NVJPEG_STATUS_EXECUTION_FAILED";
	case NVJPEG_STATUS_ARCH_MISMATCH: return "NVJPEG_STATUS_ARCH_MISMATCH";
	case NVJPEG_STATUS_INTERNAL_ERROR: return "NVJPEG_STATUS_INTERNAL_ERROR";
	case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
	default: return "Unknown nvJPEG error";
	}
}


auto rfkt::nvjpeg::encoder::encode_image(CUdeviceptr image, int width, int height, int quality, CUstream stream) -> std::future<std::vector<std::byte>>
{
	auto state = wrap_state([this, stream]() {
		std::lock_guard states_lock{ this->states_mutex };
		if (available_states.size() == 0) return make_state(stream);
		else {
			auto ret = available_states.front();
			available_states.pop();
			return ret;
		}
		}());

	nvjpegImage_t nv_image;
	nv_image.channel[0] = (unsigned char*)image;
	nv_image.pitch[0] = width;

	nv_image.channel[1] = ((unsigned char*)image) + width * height;
	nv_image.pitch[1] = width;

	nv_image.channel[2] = ((unsigned char*)image) + width * height * 2;
	nv_image.pitch[2] = width;

	std::unique_ptr<std::size_t> length_ptr = std::make_unique<std::size_t>();

	struct stream_state_t {
		decltype(state) state;
		decltype(length_ptr) length_ptr;
		std::promise<std::vector<std::byte>> promise;
		std::vector<std::byte> jpeg;
		std::size_t max_length;
	};

	auto ss = new stream_state_t{
		.state = std::move(state),
		.length_ptr = std::move(length_ptr)
	};

	auto fut = ss->promise.get_future();

	nvjpegEncodeGetBufferSize(nv_handle, params_map[quality], width, height, &ss->max_length);
	ss->jpeg.resize(ss->max_length);

	NVJPEG_SAFE_CALL(nvjpegEncodeImage(nv_handle, ss->state.get(), params_map[quality], &nv_image, NVJPEG_INPUT_RGB, width, height, stream));
	NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(nv_handle, ss->state.get(), nullptr, ss->length_ptr.get(), stream));
	NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(nv_handle, ss->state.get(), (unsigned char*) ss->jpeg.data(), ss->length_ptr.get(), stream));

	cuLaunchHostFunc(stream, [](void* d) {
		auto ss = (stream_state_t*)d;

		SPDLOG_INFO("max {}, actual {}", ss->max_length, *ss->length_ptr);
		ss->jpeg.resize(*ss->length_ptr);
		ss->promise.set_value(std::move(ss->jpeg));
		delete ss;


	}, ss);

	return fut;
}

rfkt::nvjpeg::encoder::encoder(CUstream stream)
{
	NVJPEG_SAFE_CALL(nvjpegCreateSimple(&nv_handle));

	for (int i = 0; i < max_states; i++) available_states.push(make_state(stream));
	for (unsigned char i = 0; i <= 100; i++) params_map[i] = make_params(i, stream);
}

rfkt::nvjpeg::encoder::~encoder()
{
}

auto destroy_state(nvjpegEncoderState_t state) -> nvjpegStatus_t
{
	return nvjpegEncoderStateDestroy(state);
}

auto rfkt::nvjpeg::encoder::wrap_state(nvjpegEncoderState_t state) -> std::unique_ptr<nvjpegEncoderState, std::function<void(nvjpegEncoderState_t)>>
{
	return {
		state,
		[this](nvjpegEncoderState_t state) {
			std::lock_guard states_lock{ states_mutex };
			if (available_states.size() >= max_states) {
				destroy_state(state);
			}
			else {
				available_states.push(state);
			}
		} 
	};
}

auto rfkt::nvjpeg::encoder::make_state(CUstream stream) -> nvjpegEncoderState_t
{
	nvjpegEncoderState_t state;
	nvjpegEncoderStateCreate(nv_handle, &state, stream);
	return state;
}

auto rfkt::nvjpeg::encoder::make_params(unsigned char quality, CUstream stream) -> nvjpegEncoderParams_t
{
	if (quality > 100) quality = 100;

	nvjpegEncoderParams_t ret;
	NVJPEG_SAFE_CALL(nvjpegEncoderParamsCreate(nv_handle, &ret, stream));
	NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetSamplingFactors(ret, NVJPEG_CSS_444, stream));
	nvjpegEncoderParamsSetQuality(ret, quality, stream);

	return ret;
}
