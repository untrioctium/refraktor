#include <spdlog/spdlog.h>
#include <dylib.hpp>
#include <optional>

#include <librefrakt/util/nvjpeg.h>

using nvjpegStatus_t = int;

#define NVJPEG_STATUS_SUCCESS 0

#define NVJPEG_SAFE_CALL(x)                                         \
  do {                                                            \
    nvjpegStatus_t result = x;                                          \
    if (result != NVJPEG_STATUS_SUCCESS) {                                 \
      SPDLOG_ERROR("`{}` failed with result: {}", #x, result);       \
      exit(1);                                                    \
    }                                                             \
  } while(0)

/*const char* get_nvjpeg_error_string(nvjpegStatus_t status) {
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
}*/

#define NVJPEG_MAX_COMPONENT 4

struct nvjpegImage_t
{
	unsigned char* channel[NVJPEG_MAX_COMPONENT];
	size_t    pitch[NVJPEG_MAX_COMPONENT];
};

typedef enum
{
	NVJPEG_CSS_444 = 0,
	NVJPEG_CSS_422 = 1,
	NVJPEG_CSS_420 = 2,
	NVJPEG_CSS_440 = 3,
	NVJPEG_CSS_411 = 4,
	NVJPEG_CSS_410 = 5,
	NVJPEG_CSS_GRAY = 6,
	NVJPEG_CSS_410V = 7,
	NVJPEG_CSS_UNKNOWN = -1
} nvjpegChromaSubsampling_t;

typedef enum
{
	NVJPEG_INPUT_RGB = 3, // Input is RGB - will be converted to YCbCr before encoding
	NVJPEG_INPUT_BGR = 4, // Input is RGB - will be converted to YCbCr before encoding
	NVJPEG_INPUT_RGBI = 5, // Input is interleaved RGB - will be converted to YCbCr before encoding
	NVJPEG_INPUT_BGRI = 6  // Input is interleaved RGB - will be converted to YCbCr before encoding
} nvjpegInputFormat_t;

static nvjpegStatus_t(*nvjpegCreateSimple)(nvjpegHandle_t* handle);
static nvjpegStatus_t(*nvjpegEncoderStateCreate)(nvjpegHandle_t handle, nvjpegEncoderState_t* state, RUstream stream);
static nvjpegStatus_t(*nvjpegEncoderStateDestroy)(nvjpegEncoderState_t state);
static nvjpegStatus_t(*nvjpegEncoderParamsCreate)(nvjpegHandle_t handle, nvjpegEncoderParams_t* params, RUstream stream);
static nvjpegStatus_t(*nvjpegEncoderParamsSetSamplingFactors)(nvjpegEncoderParams_t params, nvjpegChromaSubsampling_t subsampling, RUstream stream);
static nvjpegStatus_t(*nvjpegEncoderParamsSetQuality)(nvjpegEncoderParams_t params, int quality, RUstream stream);
static nvjpegStatus_t(*nvjpegEncodeImage)(nvjpegHandle_t handle, nvjpegEncoderState_t state, nvjpegEncoderParams_t params, nvjpegImage_t* image, nvjpegInputFormat_t format, int width, int height, RUstream stream);
static nvjpegStatus_t(*nvjpegEncodeRetrieveBitstream)(nvjpegHandle_t handle, nvjpegEncoderState_t state, unsigned char* pBitstream, size_t* pSize, RUstream stream);

std::optional<dylib> nvjpeg_api;

void rfkt::nvjpeg::init()
{
	if(nvjpeg_api.has_value()) return;

	nvjpeg_api = dylib{ "nvjpeg64_12" };
	auto& lib = nvjpeg_api.value();

	nvjpegCreateSimple = lib.get_function<nvjpegStatus_t(nvjpegHandle_t*)>("nvjpegCreateSimple");
	nvjpegEncoderStateCreate = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderState_t*, RUstream)>("nvjpegEncoderStateCreate");
	nvjpegEncoderStateDestroy = lib.get_function<nvjpegStatus_t(nvjpegEncoderState_t)>("nvjpegEncoderStateDestroy");
	nvjpegEncoderParamsCreate = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderParams_t*, RUstream)>("nvjpegEncoderParamsCreate");
	nvjpegEncoderParamsSetSamplingFactors = lib.get_function<nvjpegStatus_t(nvjpegEncoderParams_t, nvjpegChromaSubsampling_t, RUstream)>("nvjpegEncoderParamsSetSamplingFactors");
	nvjpegEncoderParamsSetQuality = lib.get_function<nvjpegStatus_t(nvjpegEncoderParams_t, int, RUstream)>("nvjpegEncoderParamsSetQuality");
	nvjpegEncodeImage = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderState_t, nvjpegEncoderParams_t, nvjpegImage_t*, nvjpegInputFormat_t, int, int, RUstream)>("nvjpegEncodeImage");
	nvjpegEncodeRetrieveBitstream = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char*, size_t*, RUstream)>("nvjpegEncodeRetrieveBitstream");
}

auto rfkt::nvjpeg::encoder::encode_image(RUdeviceptr image, int width, int height, int quality, RUstream stream) -> std::future<std::move_only_function<std::vector<std::byte>()>>
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


	struct stream_state_t {
		decltype(state) state;
		decltype(nv_handle) handle;
		std::promise<std::move_only_function<std::vector<std::byte>()>> promise;
	};

	auto ss = new stream_state_t{
		.state = std::move(state),
		.handle = nv_handle
	};

	auto fut = ss->promise.get_future();

	NVJPEG_SAFE_CALL(nvjpegEncodeImage(nv_handle, ss->state.get(), params_map[quality], &nv_image, NVJPEG_INPUT_RGB, width, height, stream));

	ruLaunchHostFunc(stream, [](void* d) {
		auto ss = (stream_state_t*)d;

		auto func = [state = std::move(ss->state), handle = ss->handle]() -> std::vector<std::byte> {

			std::size_t length;

			NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(handle, state.get(), nullptr, &length, nullptr));

			std::vector<std::byte> ret(length);

			NVJPEG_SAFE_CALL(nvjpegEncodeRetrieveBitstream(handle, state.get(), (unsigned char*) ret.data(), &length, nullptr));
			return ret;
		};

		ss->promise.set_value(std::move(func));
		delete ss;

	}, ss);

	return fut;
}

rfkt::nvjpeg::encoder::encoder(RUstream stream)
{
	/*dev_alloc = {
		[](void** ptr, std::size_t size) -> int {
			auto result = cuMemAlloc((CUdeviceptr*)ptr, size);
			if (result != CUDA_SUCCESS) __debugbreak();
			return (result == CUDA_SUCCESS) ? 0: 1;
		},
		[](void* ptr) -> int {
			return (cuMemFree((CUdeviceptr)ptr) == CUDA_SUCCESS) ? 0: 1;
		}
	};

	p_alloc = {
		[](void** ptr, std::size_t size, unsigned int flags) -> int {
			auto result = cuMemAllocHost(ptr, size);
			if (result != CUDA_SUCCESS) __debugbreak();
			return (result == CUDA_SUCCESS) ? 0 : 1;
		},
		[](void* ptr) -> int {
			return (cuMemFreeHost(ptr) == CUDA_SUCCESS) ? 0 : 1;
		}
	};

	nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_alloc, &p_alloc, 0, &nv_handle);
	*/
	NVJPEG_SAFE_CALL(nvjpegCreateSimple(&nv_handle));

	for (int i = 0; i < max_states; i++) available_states.push(make_state(stream));
	for (unsigned char i = 1; i <= 100; i++) params_map[i] = make_params(i, stream);
}

rfkt::nvjpeg::encoder::~encoder()
{
}

auto rfkt::nvjpeg::encoder::wrap_state(nvjpegEncoderState_t state) -> std::unique_ptr<nvjpegEncoderState, std::function<void(nvjpegEncoderState_t)>>
{
	return {
		state,
		[this](nvjpegEncoderState_t state) {
			std::lock_guard states_lock{ states_mutex };
			if (available_states.size() >= max_states) {
				nvjpegEncoderStateDestroy(state);
			}
			else {
				available_states.push(state);
			}
		} 
	};
}

auto rfkt::nvjpeg::encoder::make_state(RUstream stream) -> nvjpegEncoderState_t
{
	SPDLOG_INFO("Creating state");
	nvjpegEncoderState_t state;
	NVJPEG_SAFE_CALL(nvjpegEncoderStateCreate(nv_handle, &state, stream));
	return state;
}

auto rfkt::nvjpeg::encoder::make_params(unsigned char quality, RUstream stream) -> nvjpegEncoderParams_t
{
	if (quality > 100) quality = 100;

	nvjpegEncoderParams_t ret;
	NVJPEG_SAFE_CALL(nvjpegEncoderParamsCreate(nv_handle, &ret, stream));
	NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetSamplingFactors(ret, NVJPEG_CSS_444, stream));
	NVJPEG_SAFE_CALL(nvjpegEncoderParamsSetQuality(ret, quality, stream));

	return ret;
}
