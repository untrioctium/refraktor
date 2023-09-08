#include <unordered_map>
#include <format>

#include <ffnvcodec/nvEncodeAPI.h>
#include <dylib.hpp>

#include <eznve.hpp>
#include <iostream>


#define CHECK_NVENC(expr) \
do { \
	if(auto ret = expr; ret != NV_ENC_SUCCESS) { \
		std::string error = std::format("{} failed with {} ({}@{})", #expr, get_error(ret), __FILE__, __LINE__); \
		std::cerr << api.funcs().nvEncGetLastErrorString(session) << std::endl; \
		std::cerr << error << std::endl; \
		__debugbreak(); \
		throw std::runtime_error{ error }; \
	} \
} while(0) \

using nv_create_api = NVENCSTATUS(NVENCAPI)(NV_ENCODE_API_FUNCTION_LIST*);
using nv_check_version = NVENCSTATUS(NVENCAPI)(uint32_t*);

consteval bool is_posix() {
#ifdef _WIN32
	return false;
#else
	return true;
#endif
}

std::string_view get_error(NVENCSTATUS code) {
	static const std::unordered_map<NVENCSTATUS, std::string> codes{
		{NV_ENC_SUCCESS, "SUCCESS"},
		{NV_ENC_ERR_NO_ENCODE_DEVICE, "NO_ENCODE_DEVICE"},
		{NV_ENC_ERR_UNSUPPORTED_DEVICE, "UNSUPPORTED_DEVICE"},
		{NV_ENC_ERR_INVALID_ENCODERDEVICE, "INVALID_ENCODERDEVICE"},
		{NV_ENC_ERR_INVALID_DEVICE, "INVALID_DEVICE"},
		{NV_ENC_ERR_DEVICE_NOT_EXIST, "DEVICE_NOT_EXIST"},
		{NV_ENC_ERR_INVALID_PTR, "INVALID_PTR"},
		{NV_ENC_ERR_INVALID_EVENT, "INVALID_EVENT"},
		{NV_ENC_ERR_INVALID_PARAM, "INVALID_PARAM"},
		{NV_ENC_ERR_INVALID_CALL, "INVALID_CALL"},
		{NV_ENC_ERR_OUT_OF_MEMORY, "OUT_OF_MEMORY"},
		{NV_ENC_ERR_ENCODER_NOT_INITIALIZED, "ENCODER_NOT_INITIALIZED"},
		{NV_ENC_ERR_UNSUPPORTED_PARAM, "UNSUPPORTED_PARAM"},
		{NV_ENC_ERR_LOCK_BUSY, "LOCK_BUSY"},
		{NV_ENC_ERR_NOT_ENOUGH_BUFFER, "NOT_ENOUGH_BUFFER"},
		{NV_ENC_ERR_INVALID_VERSION, "INVALID_VERSION"},
		{NV_ENC_ERR_MAP_FAILED, "MAP_FAILED"},
		{NV_ENC_ERR_NEED_MORE_INPUT, "NEED_MORE_INPUT"},
		{NV_ENC_ERR_ENCODER_BUSY, "ENCODER_BUSY"},
		{NV_ENC_ERR_EVENT_NOT_REGISTERD, "EVENT_NOT_REGISTERD"},
		{NV_ENC_ERR_GENERIC, "GENERIC"},
		{NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY, "INCOMPATIBLE_CLIENT_KEY"},
		{NV_ENC_ERR_UNIMPLEMENTED, "UNIMPLEMENTED"},
		{NV_ENC_ERR_RESOURCE_REGISTER_FAILED, "RESOURCE_REGISTER_FAILED"},
		{NV_ENC_ERR_RESOURCE_NOT_REGISTERED, "RESOURCE_NOT_REGISTERED"},
		{NV_ENC_ERR_RESOURCE_NOT_MAPPED, "RESOURCE_NOT_MAPPED"}
	};

	return codes.at(code);
}

class api_t {
public:
	api_t() {
		auto create_api = lib_.get_function<nv_create_api>("NvEncodeAPICreateInstance");
		auto check_version = lib_.get_function<nv_check_version>("NvEncodeAPIGetMaxSupportedVersion");

		uint32_t header_version = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
		uint32_t device_version = 0;
		check_version(&device_version);
		if (header_version > device_version) throw std::runtime_error{ "unsupported nvenc version" };

		funcs_.version = NV_ENCODE_API_FUNCTION_LIST_VER;
		if (create_api(&funcs_) != NV_ENC_SUCCESS) {
			throw std::runtime_error{ "could not create nvenc API" };
		}
	}

	const auto& funcs() const { return funcs_; }

private:
	dylib lib_{ is_posix() ? "libnvidia-encode.so" : "nvEncodeAPI64", false };
	NV_ENCODE_API_FUNCTION_LIST funcs_;
};

inline static const auto api = api_t{};

eznve::encoder::encoder(uint2 dims, uint2 fps, codec c, CUcontext ctx) : dims(dims) {
	const auto& funcs = api.funcs();

	auto session_params = pbuf_as<NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS>();
	session_params->apiVersion = NVENCAPI_VERSION;
	session_params->deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	session_params->device = ctx;
	session_params->version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
	CHECK_NVENC(funcs.nvEncOpenEncodeSessionEx(session_params, &session));

	uint32_t guid_count = 0;
	funcs.nvEncGetEncodeGUIDCount(session, &guid_count);
	std::vector<GUID> guids(guid_count);
	funcs.nvEncGetEncodeGUIDs(session, guids.data(), guid_count, &guid_count);

	uint32_t preset_count = 0;
	funcs.nvEncGetEncodePresetCount(session, guids[1], &preset_count);
	std::vector<GUID> presets(preset_count);
	funcs.nvEncGetEncodePresetGUIDs(session, guids[1], presets.data(), preset_count, &preset_count);

	NV_ENC_INITIALIZE_PARAMS init_params{};
	NV_ENC_CONFIG encoder_config{};

	init_params.encodeConfig = &encoder_config;
	encoder_config.version = NV_ENC_CONFIG_VER;
	init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;

	init_params.encodeGUID = (c == codec::h264)? NV_ENC_CODEC_H264_GUID : (c == codec::hevc) ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_AV1_GUID;
	init_params.presetGUID = NV_ENC_PRESET_P7_GUID;
	init_params.encodeWidth = dims.x;
	init_params.encodeHeight = dims.y;
	init_params.darWidth = dims.x;
	init_params.darHeight = dims.y;
	init_params.frameRateNum = fps.x;
	init_params.frameRateDen = fps.y;
	init_params.enableEncodeAsync = 0;
	init_params.enablePTD = 1;
	init_params.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

	NV_ENC_PRESET_CONFIG preset_config = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER } };
	CHECK_NVENC(funcs.nvEncGetEncodePresetConfigEx(session, init_params.encodeGUID, init_params.presetGUID, init_params.tuningInfo, &preset_config));


	memcpy(init_params.encodeConfig, &preset_config.presetCfg, sizeof(NV_ENC_CONFIG));

	auto& hevc_cfg = init_params.encodeConfig->encodeCodecConfig.hevcConfig;

	init_params.encodeConfig->profileGUID = NV_ENC_HEVC_PROFILE_MAIN10_GUID;
	hevc_cfg.pixelBitDepthMinus8 = 2;
	hevc_cfg.chromaFormatIDC = 3;
	hevc_cfg.hevcVUIParameters.videoSignalTypePresentFlag = 1;
	hevc_cfg.hevcVUIParameters.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_COMPONENT;
	hevc_cfg.hevcVUIParameters.colourDescriptionPresentFlag = 1;
	hevc_cfg.hevcVUIParameters.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT2020;
	hevc_cfg.hevcVUIParameters.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE2084;
	hevc_cfg.hevcVUIParameters.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT2020_NCL;
	hevc_cfg.hevcVUIParameters.videoFullRangeFlag = 1;

	CHECK_NVENC(funcs.nvEncInitializeEncoder(session, &init_params));

	for (int i = 0; i < encoder_config.frameIntervalP; i++) {
		push_buffer();
	}
}

eznve::encoder::~encoder() {
	const auto& funcs = api.funcs();

	if (frames_encoded > 0) 
		try {
			flush();
		} catch (...) {}

	for (auto& buf : buffers) {
		funcs.nvEncDestroyBitstreamBuffer(session, buf.out_stream);
		funcs.nvEncUnregisterResource(session, buf.registration);
		cuMemFree(buf.ptr);
	}

	funcs.nvEncDestroyEncoder(session);
}

std::vector<eznve::chunk> eznve::encoder::submit_frame(frame_flag flag) {
	const auto& funcs = api.funcs();

	auto& buf = buffers[current_buffer];

	buf.map(session);

	auto pic_params = pbuf_as<NV_ENC_PIC_PARAMS>();
	pic_params->version = NV_ENC_PIC_PARAMS_VER;
	pic_params->bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR10;
	pic_params->pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	pic_params->inputBuffer = buf.mapped;
	pic_params->outputBitstream = buf.out_stream;
	pic_params->inputPitch = dims.x * 4;

	if(flag == frame_flag::idr || frames_encoded == 0) {
		pic_params->encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
	}

	if(frames_encoded == 0) {
		pic_params->encodePicFlags |= NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
	}

	auto frame_status = funcs.nvEncEncodePicture(session, pic_params);
	frames_encoded++;

	std::vector<chunk> chunks;

	if (frame_status == NV_ENC_ERR_NEED_MORE_INPUT) {
		current_buffer++;
		if (current_buffer >= buffers.size()) push_buffer();
		return chunks;
	}
	CHECK_NVENC(frame_status);
	std::cout << "pushing " << current_buffer + 1 << " frames" << std::endl;
	for (int i = 0; i <= current_buffer; i++) {
		auto chunk = buffers[i].lock_stream(session);
		std::cout << "pushing " << chunk.data.size() << " bytes" << std::endl;
		bytes_encoded += chunk.data.size();
		chunks.emplace_back(std::move(chunk));
		buffers[i].unlock_stream(session);
		buffers[i].unmap(session);
	}

	current_buffer = 0;
	return chunks;
}

std::vector<eznve::chunk> eznve::encoder::flush() {
	bytes_encoded = 0;
	frames_encoded = 0;

	auto pic_params = pbuf_as<NV_ENC_PIC_PARAMS>();
	pic_params->version = NV_ENC_PIC_PARAMS_VER;
	pic_params->encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	pic_params->outputBitstream = buffers[current_buffer].out_stream;

	auto frame_status = api.funcs().nvEncEncodePicture(session, pic_params);

	std::vector<chunk> chunks;
	if (frame_status == NV_ENC_ERR_NEED_MORE_INPUT) return chunks;
	CHECK_NVENC(frame_status);

	for (int i = 0; i <= current_buffer; i++) {
		auto chunk = buffers[i].lock_stream(session);\
		std::cout << "pushing " << chunk.data.size() << " bytes" << std::endl;
		//bytes_encoded += chunk.data.size();
		chunks.emplace_back(std::move(chunk));
		buffers[i].unlock_stream(session);
		if(i != current_buffer) buffers[i].unmap(session);
	}

	current_buffer = 0;
	return chunks;
}

void eznve::encoder::push_buffer()
{
	auto buf = buffer_t{};

	cuMemAlloc(&buf.ptr, dims.x * dims.y * 4);

	auto out_buf = pbuf_as<NV_ENC_CREATE_BITSTREAM_BUFFER>();
	out_buf->version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
	CHECK_NVENC(api.funcs().nvEncCreateBitstreamBuffer(session, out_buf));
	buf.out_stream = out_buf->bitstreamBuffer;

	auto input_res = pbuf_as<NV_ENC_REGISTER_RESOURCE>();
	input_res->version = NV_ENC_REGISTER_RESOURCE_VER;
	input_res->resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
	input_res->width = dims.x;
	input_res->height = dims.y;
	input_res->pitch = dims.x * 4;
	input_res->subResourceIndex = 0;
	input_res->resourceToRegister = (void*)buf.ptr;
	input_res->bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR10;
	input_res->bufferUsage = NV_ENC_INPUT_IMAGE;
	CHECK_NVENC(api.funcs().nvEncRegisterResource(session, input_res));
	buf.registration = input_res->registeredResource;

	buffers.emplace_back(std::move(buf));
}

void eznve::encoder::buffer_t::map(void* session)
{
	NV_ENC_MAP_INPUT_RESOURCE map;
	std::memset(&map, 0, sizeof(map));

	map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
	map.registeredResource = registration;
	CHECK_NVENC(api.funcs().nvEncMapInputResource(session, &map));

	mapped = map.mappedResource;
}

void eznve::encoder::buffer_t::unmap(void* session)
{
	CHECK_NVENC(api.funcs().nvEncUnmapInputResource(session, mapped));
	mapped = nullptr;
}

eznve::chunk eznve::encoder::buffer_t::lock_stream(void* session)
{
	NV_ENC_LOCK_BITSTREAM lock;
	lock.version = NV_ENC_LOCK_BITSTREAM_VER;
	lock.outputBitstream = out_stream;
	lock.doNotWait = 0;
	//std::cout << "locking " << out_stream << std::endl;
	CHECK_NVENC(api.funcs().nvEncLockBitstream(session, &lock));

	auto chunk_span = std::span<const char>{ (const char*)lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes };

	return eznve::chunk{
		.data = {chunk_span.begin(), chunk_span.end()},
		.index = lock.frameIdx,
		.timestamp = lock.outputTimeStamp,
		.duration = lock.outputDuration
	};
}

void eznve::encoder::buffer_t::unlock_stream(void* session)
{
	CHECK_NVENC(api.funcs().nvEncUnlockBitstream(session, out_stream));
}
