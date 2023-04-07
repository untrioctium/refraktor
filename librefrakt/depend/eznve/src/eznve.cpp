#include <unordered_map>
#include <format>

#include <ffnvcodec/nvEncodeAPI.h>
#include <dylib.hpp>

#include <eznve.hpp>


#define CHECK_NVENC(expr) \
do { \
	if(auto ret = expr; ret != NV_ENC_SUCCESS) { \
		__debugbreak(); \
		throw std::runtime_error{ \
			std::format("{} failed with {} ({}@{})", #expr, get_error(ret), __FILE__, __LINE__) \
		}; \
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

	cuMemAlloc(&input_buffer, dims.x * dims.y * 4);

	auto session_params = pbuf_as<NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS>();
	session_params->apiVersion = NVENCAPI_VERSION;
	session_params->deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	session_params->device = ctx;
	session_params->version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
	CHECK_NVENC(funcs.nvEncOpenEncodeSessionEx(session_params, &session));

	struct config_package_t {
		NV_ENC_INITIALIZE_PARAMS init_params;
		NV_ENC_PRESET_CONFIG preset_config;
	};

	auto config_package = pbuf_as<config_package_t>();

	auto& init_params = config_package->init_params;
	init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
	init_params.encodeGUID = (c == codec::h264)? NV_ENC_CODEC_H264_GUID : NV_ENC_CODEC_HEVC_GUID;
	init_params.presetGUID = NV_ENC_PRESET_HQ_GUID;
	init_params.encodeWidth = dims.x;
	init_params.encodeHeight = dims.y;
	init_params.darWidth = dims.x;
	init_params.darHeight = dims.y;
	init_params.frameRateNum = fps.x;
	init_params.frameRateDen = fps.y;
	init_params.enableEncodeAsync = 0;
	init_params.enablePTD = 1;

	auto& preset_config = config_package->preset_config;
	preset_config.version = NV_ENC_PRESET_CONFIG_VER;
	preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
	CHECK_NVENC(funcs.nvEncGetEncodePresetConfig(session, init_params.encodeGUID, init_params.presetGUID, &preset_config));
	init_params.encodeConfig = &preset_config.presetCfg;

	CHECK_NVENC(funcs.nvEncInitializeEncoder(session, &init_params));

	auto out_buf = pbuf_as<NV_ENC_CREATE_BITSTREAM_BUFFER>();
	out_buf->version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
	CHECK_NVENC(funcs.nvEncCreateBitstreamBuffer(session, out_buf));
	out_stream = out_buf->bitstreamBuffer;

	auto input_res = pbuf_as<NV_ENC_REGISTER_RESOURCE>();
	input_res->version = NV_ENC_REGISTER_RESOURCE_VER;
	input_res->resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
	input_res->width = dims.x;
	input_res->height = dims.y;
	input_res->pitch = dims.x * 4;
	input_res->subResourceIndex = 0;
	input_res->resourceToRegister = (void*)input_buffer;
	input_res->bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
	input_res->bufferUsage = NV_ENC_INPUT_IMAGE;
	CHECK_NVENC(funcs.nvEncRegisterResource(session, input_res));
	in_registration = input_res->registeredResource;
}

eznve::encoder::~encoder() {
	const auto& funcs = api.funcs();

	if(frames_encoded > 0) flush();

	funcs.nvEncDestroyBitstreamBuffer(session, out_stream);
	funcs.nvEncUnregisterResource(session, in_registration);
	funcs.nvEncDestroyEncoder(session);
	cuMemFree(input_buffer);
}

class input_resource_mapper {
public:
	input_resource_mapper(void* session, void* in_reg, NV_ENC_MAP_INPUT_RESOURCE* in_map) : session(session), in_reg(in_reg) {
		in_map->version = NV_ENC_MAP_INPUT_RESOURCE_VER;
		in_map->registeredResource = in_reg;
		CHECK_NVENC(api.funcs().nvEncMapInputResource(session, in_map));

		mapped = in_map->mappedResource;
	}

	~input_resource_mapper() {
		if(session && mapped) api.funcs().nvEncUnmapInputResource(session, mapped);
	}

	input_resource_mapper(const input_resource_mapper&) = delete;
	input_resource_mapper& operator=(const input_resource_mapper&) = delete;

	input_resource_mapper& operator=(input_resource_mapper&& o) noexcept {
		std::swap(mapped, o.mapped);
		std::swap(session, o.session);
		std::swap(in_reg, o.in_reg);
		return *this;
	}

	input_resource_mapper(input_resource_mapper&& o) noexcept {
		(*this) = std::move(o);
	}

	auto get() { return mapped; }

private:
	NV_ENC_INPUT_PTR mapped = nullptr;
	void* session = nullptr;
	void* in_reg = nullptr;
};

class output_bitstream_mapper {
public:
	output_bitstream_mapper(void* session, void* out_stream, NV_ENC_LOCK_BITSTREAM* mapped) : session(session), out_stream(out_stream) {
		
		mapped->version = NV_ENC_LOCK_BITSTREAM_VER;
		mapped->outputBitstream = out_stream;
		CHECK_NVENC(api.funcs().nvEncLockBitstream(session, mapped));

		auto chunk_span = std::span<const char>{ (const char*)mapped->bitstreamBufferPtr, mapped->bitstreamSizeInBytes };

		info = eznve::chunk{
			.data = {chunk_span.begin(), chunk_span.end()},
			.index = mapped->frameIdx,
			.timestamp = mapped->outputTimeStamp,
			.duration = mapped->outputDuration
		};
	}

	~output_bitstream_mapper() {
		if (session && out_stream) api.funcs().nvEncUnlockBitstream(session, out_stream);
	}

	auto& get() {
		return info;
	}

private:
	eznve::chunk info;
	void* out_stream = nullptr;
	void* session = nullptr;
};

std::optional<eznve::chunk> eznve::encoder::submit_frame(frame_flag flag) {
	const auto& funcs = api.funcs();

	input_resource_mapper res_map{ session, in_registration, pbuf_as<NV_ENC_MAP_INPUT_RESOURCE>()};

	auto pic_params = pbuf_as<NV_ENC_PIC_PARAMS>();
	pic_params->version = NV_ENC_PIC_PARAMS_VER;
	pic_params->bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
	pic_params->pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	pic_params->inputBuffer = res_map.get();
	pic_params->outputBitstream = out_stream;

	if(flag == frame_flag::idr || frames_encoded == 0) {
		pic_params->encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
	}

	if(frames_encoded == 0) {
		pic_params->encodePicFlags |= NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
	}

	auto frame_status = funcs.nvEncEncodePicture(session, pic_params);
	frames_encoded++;

	if (frame_status == NV_ENC_ERR_NEED_MORE_INPUT) return std::nullopt;
	CHECK_NVENC(frame_status);

	output_bitstream_mapper out_map{session, out_stream, pbuf_as<NV_ENC_LOCK_BITSTREAM>()};
	bytes_encoded += out_map.get().data.size();
	return std::move(out_map.get());
}

std::optional<eznve::chunk> eznve::encoder::flush() {
	bytes_encoded = 0;
	frames_encoded = 0;

	auto pic_params = pbuf_as<NV_ENC_PIC_PARAMS>();
	pic_params->version = NV_ENC_PIC_PARAMS_VER;
	pic_params->encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	pic_params->outputBitstream = out_stream;

	auto frame_status = api.funcs().nvEncEncodePicture(session, pic_params);

	if (frame_status == NV_ENC_ERR_NEED_MORE_INPUT) return std::nullopt;
	CHECK_NVENC(frame_status);

	output_bitstream_mapper out_map{ session, out_stream, pbuf_as<NV_ENC_LOCK_BITSTREAM>() };
	return std::move(out_map.get());
}