#include <spdlog/spdlog.h>
#include <dylib.hpp>
#include <queue>

#include <librefrakt/interface/jpeg_encoder.h>

constexpr static std::size_t NVJPEG_MAX_COMPONENT = 4;
constexpr static int NVJPEG_STATUS_SUCCESS = 0;

#define NVJPEG_SAFE_CALL(x)                                       \
  do {                                                            \
    nvjpegStatus_t result = x;                                    \
    if (result != NVJPEG_STATUS_SUCCESS) {                        \
      SPDLOG_ERROR("`{}` failed with result: {}", #x, result);    \
      __debugbreak();                                             \
      exit(1);                                                    \
    }                                                             \
  } while(0)

using nvjpegStatus_t = int;
using nvjpegHandle_t = struct nvJpegHandle*;
using nvjpegEncoderState_t = struct nvjpegEncoderState*;
using nvjpegEncoderParams_t = struct nvJpegEncoderParams_st*;

struct nvjpegImage_t
{
	std::array<unsigned char*, NVJPEG_MAX_COMPONENT> channel;
	std::array<std::size_t, NVJPEG_MAX_COMPONENT> pitch;
};

enum nvjpegChromaSubsampling_t { NVJPEG_CSS_444 = 0 };
enum nvjpegInputFormat_t { NVJPEG_INPUT_RGBI = 5 };
enum nvjpegBackend_t { NVJPEG_BACKEND_DEFAULT = 0 };

typedef RUresult (*tDevMalloc)(RUdeviceptr*, size_t);
typedef RUresult(*tDevFree)(RUdeviceptr);

typedef RUresult(*tPinnedMalloc)(void**, size_t, unsigned int flags);
typedef RUresult(*tPinnedFree)(void*);

typedef struct
{
	tDevMalloc dev_malloc;
	tDevFree dev_free;
} nvjpegDevAllocator_t;

typedef struct
{
	tPinnedMalloc pinned_malloc;
	tPinnedFree pinned_free;
} nvjpegPinnedAllocator_t;

struct nvjpeg_api_table {

	explicit nvjpeg_api_table(dylib&& dlib) : lib{ std::move(dlib) } {
		nvjpegCreateSimple = lib.get_function<nvjpegStatus_t(nvjpegHandle_t*)>("nvjpegCreateSimple");
		nvjpegCreateEx = lib.get_function<nvjpegStatus_t(nvjpegBackend_t, nvjpegDevAllocator_t*, nvjpegPinnedAllocator_t*, unsigned int, nvjpegHandle_t*)>("nvjpegCreateEx");
		nvjpegDestroy = lib.get_function<nvjpegStatus_t(nvjpegHandle_t)>("nvjpegDestroy");
		nvjpegEncoderStateCreate = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderState_t*, RUstream)>("nvjpegEncoderStateCreate");
		nvjpegEncoderStateDestroy = lib.get_function<nvjpegStatus_t(nvjpegEncoderState_t)>("nvjpegEncoderStateDestroy");
		nvjpegEncoderParamsCreate = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderParams_t*, RUstream)>("nvjpegEncoderParamsCreate");
		nvjpegEncoderParamsSetSamplingFactors = lib.get_function<nvjpegStatus_t(nvjpegEncoderParams_t, nvjpegChromaSubsampling_t, RUstream)>("nvjpegEncoderParamsSetSamplingFactors");
		nvjpegEncoderParamsSetQuality = lib.get_function<nvjpegStatus_t(nvjpegEncoderParams_t, int, RUstream)>("nvjpegEncoderParamsSetQuality");
		nvjpegEncodeImage = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderState_t, nvjpegEncoderParams_t, nvjpegImage_t*, nvjpegInputFormat_t, int, int, RUstream)>("nvjpegEncodeImage");
		nvjpegEncodeRetrieveBitstream = lib.get_function<nvjpegStatus_t(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char*, size_t*, RUstream)>("nvjpegEncodeRetrieveBitstream");
	
	}

	dylib lib;

	nvjpegStatus_t(*nvjpegCreateSimple)(nvjpegHandle_t* handle);
	nvjpegStatus_t(*nvjpegCreateEx)(nvjpegBackend_t backend, nvjpegDevAllocator_t* dev_allocator, nvjpegPinnedAllocator_t* pinned_allocator, unsigned int flags, nvjpegHandle_t* handle);
	nvjpegStatus_t(*nvjpegDestroy)(nvjpegHandle_t handle);
	nvjpegStatus_t(*nvjpegEncoderStateCreate)(nvjpegHandle_t handle, nvjpegEncoderState_t* state, RUstream stream);
	nvjpegStatus_t(*nvjpegEncoderStateDestroy)(nvjpegEncoderState_t state);
	nvjpegStatus_t(*nvjpegEncoderParamsCreate)(nvjpegHandle_t handle, nvjpegEncoderParams_t* params, RUstream stream);
	nvjpegStatus_t(*nvjpegEncoderParamsSetSamplingFactors)(nvjpegEncoderParams_t params, nvjpegChromaSubsampling_t subsampling, RUstream stream);
	nvjpegStatus_t(*nvjpegEncoderParamsSetQuality)(nvjpegEncoderParams_t params, int quality, RUstream stream);
	nvjpegStatus_t(*nvjpegEncodeImage)(nvjpegHandle_t handle, nvjpegEncoderState_t state, nvjpegEncoderParams_t params, nvjpegImage_t* image, nvjpegInputFormat_t format, int width, int height, RUstream stream);
	nvjpegStatus_t(*nvjpegEncodeRetrieveBitstream)(nvjpegHandle_t handle, nvjpegEncoderState_t state, unsigned char* pBitstream, size_t* pSize, RUstream stream);
};

namespace rfkt {

	struct nvjpeg_encoder : public jpeg_encoder::registrar<nvjpeg_encoder> {
		const static inline meta_type meta = {
			.priority = 0,
			.supported_apis = { ROCCU_API_CUDA }
		};

		explicit nvjpeg_encoder(roccu::gpu_stream& stream) : api(dylib{ "nvjpeg64_12" }) {

			dev_allocator.dev_malloc = ruMemAlloc;
			dev_allocator.dev_free = ruMemFree;

			NVJPEG_SAFE_CALL(api.nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, nullptr, 0, &nv_handle));

			for (int i = 0; i < max_states; i++) available_states.push(make_state(stream));
			for (unsigned char i = 1; i <= 100; i++) params_map[i] = make_params(i, stream);
		}

		~nvjpeg_encoder() {}

		auto encode_image(roccu::gpu_image_view<uchar3> image, int quality, roccu::gpu_stream& stream) -> std::future<encode_thunk> override {
			auto state = get_or_make_state(stream);

			nvjpegImage_t nv_image;
			memset(&nv_image, 0, sizeof(nv_image));
			nv_image.channel[0] = (unsigned char*)image.ptr();
			nv_image.pitch[0] = image.pitch() * decltype(image)::element_size;

			struct stream_state_t {
				decltype(state) state;
				decltype(nv_handle) handle;
				nvjpeg_api_table* api;
				std::promise<encode_thunk> promise;
			};

			auto ss = new stream_state_t{
				.state = std::move(state),
				.handle = nv_handle,
				.api = &api
			};

			auto fut = ss->promise.get_future();

			NVJPEG_SAFE_CALL(api.nvjpegEncodeImage(nv_handle, ss->state.get(), params_map[quality], &nv_image, NVJPEG_INPUT_RGBI, image.width(), image.height(), stream));

			ruLaunchHostFunc(stream, [](void* d) {
				auto ss = (stream_state_t*)d;

				auto func = [state = std::move(ss->state), handle = ss->handle, api = ss->api] {

					std::size_t length;

					NVJPEG_SAFE_CALL(api->nvjpegEncodeRetrieveBitstream(handle, state.get(), nullptr, &length, nullptr));

					std::vector<std::byte> ret(length);

					NVJPEG_SAFE_CALL(api->nvjpegEncodeRetrieveBitstream(handle, state.get(), (unsigned char*)ret.data(), &length, nullptr));
					return ret;
				};

				ss->promise.set_value(std::move(func));
				delete ss;

			}, ss);

			return fut;

		}

		auto wrap_state(nvjpegEncoderState_t state) -> std::unique_ptr<nvjpegEncoderState, std::function<void(nvjpegEncoderState_t)>> {
			return {
				state,
				[this](nvjpegEncoderState_t state) {
					std::lock_guard states_lock{ states_mutex };
					if (available_states.size() >= max_states) {
						api.nvjpegEncoderStateDestroy(state);
					}
					else {
						available_states.push(state);
					}
				}
			};
		}

		auto make_state(roccu::gpu_stream& stream) -> nvjpegEncoderState_t {
			nvjpegEncoderState_t state;
			NVJPEG_SAFE_CALL(api.nvjpegEncoderStateCreate(nv_handle, &state, stream));
			return state;
		}

		auto make_params(unsigned char quality, roccu::gpu_stream& stream) -> nvjpegEncoderParams_t {
			if (quality > 100) quality = 100;

			nvjpegEncoderParams_t ret;
			NVJPEG_SAFE_CALL(api.nvjpegEncoderParamsCreate(nv_handle, &ret, stream));
			NVJPEG_SAFE_CALL(api.nvjpegEncoderParamsSetSamplingFactors(ret, NVJPEG_CSS_444, stream));
			NVJPEG_SAFE_CALL(api.nvjpegEncoderParamsSetQuality(ret, quality, stream));

			return ret;
		}

		auto get_or_make_state(roccu::gpu_stream& stream) -> decltype(wrap_state(std::declval<nvjpegEncoderState_t>())) {
			std::lock_guard states_lock{ states_mutex };
			if (available_states.size() == 0) return wrap_state(make_state(stream));
			else {
				auto ret = available_states.front();
				available_states.pop();
				return wrap_state(ret);
			}
		}

		std::mutex states_mutex;
		std::queue<nvjpegEncoderState_t> available_states;
		std::unordered_map<unsigned char, nvjpegEncoderParams_t> params_map;

		int max_states = 1;

		nvjpegHandle_t nv_handle;
		nvjpeg_api_table api;

		nvjpegDevAllocator_t dev_allocator;
	};

}