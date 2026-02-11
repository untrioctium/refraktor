#include <unordered_map>
#include <format>

#include <ffnvcodec/nvEncodeAPI.h>
#include <dylib.hpp>

#include <eznve.hpp>
#include <iostream>
#include <cstring>


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

using nv_create_api = NVENCSTATUS(NV_ENCODE_API_FUNCTION_LIST*);
using nv_check_version = NVENCSTATUS(uint32_t*);

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
	NV_ENCODE_API_FUNCTION_LIST funcs_{};
};

inline static const auto api = api_t{};

namespace eznve {
	void apply_config(void* session, const api_t& api, const config& cfg, NV_ENC_INITIALIZE_PARAMS& init_params, NV_ENC_CONFIG& encoder_config) {
    
		// Quality preset -> NVENC preset GUID
		switch (cfg.preset) {
			case config::quality_preset::fastest:
				init_params.presetGUID = NV_ENC_PRESET_P1_GUID;
				break;
			case config::quality_preset::fast:
				init_params.presetGUID = NV_ENC_PRESET_P3_GUID;
				break;
			case config::quality_preset::balanced:
				init_params.presetGUID = NV_ENC_PRESET_P4_GUID;
				break;
			case config::quality_preset::quality:
				init_params.presetGUID = NV_ENC_PRESET_P5_GUID;
				break;
			case config::quality_preset::high_quality:
				init_params.presetGUID = NV_ENC_PRESET_P7_GUID;
				break;
		}
		
		// Tuning -> NVENC tuning info
		switch (cfg.tune) {
			case config::tuning::high_quality:
				init_params.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
				break;
			case config::tuning::low_latency:
				init_params.tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;
				break;
			case config::tuning::ultra_low_latency:
				init_params.tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
				break;
		}
		
		// Load preset defaults first (after setting presetGUID and tuningInfo)
		NV_ENC_PRESET_CONFIG preset_config = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER } };
		api.funcs().nvEncGetEncodePresetConfigEx(session, init_params.encodeGUID, 
									  init_params.presetGUID, init_params.tuningInfo, 
									  &preset_config);
		memcpy(&encoder_config, &preset_config.presetCfg, sizeof(NV_ENC_CONFIG));
		
		// Rate control mode
		switch (cfg.rc) {
			case config::rate_control::cqp:
				encoder_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
				encoder_config.rcParams.constQP.qpInterP = cfg.cqp;
				encoder_config.rcParams.constQP.qpInterB = cfg.cqp;
				encoder_config.rcParams.constQP.qpIntra = cfg.cqp;
				break;
			case config::rate_control::vbr:
				encoder_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
				encoder_config.rcParams.averageBitRate = cfg.bitrate_kbps * 1000;
				encoder_config.rcParams.maxBitRate = cfg.bitrate_kbps * 1500; // 1.5x headroom
				break;
			case config::rate_control::cbr:
				encoder_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
				encoder_config.rcParams.averageBitRate = cfg.bitrate_kbps * 1000;
				encoder_config.rcParams.maxBitRate = cfg.bitrate_kbps * 1000;
				break;
		}
		
		// B-frames
		if (cfg.enable_bframes) {
			// Keep preset default (usually 2-3 B-frames for high quality)
		} else {
			encoder_config.frameIntervalP = 1; // 1 = no B-frames (I and P only)
		}
		
		// Lookahead
		if (cfg.enable_lookahead) {
			encoder_config.rcParams.enableLookahead = 1;
			encoder_config.rcParams.lookaheadDepth = cfg.lookahead_depth;
		} else {
			encoder_config.rcParams.enableLookahead = 0;
			encoder_config.rcParams.lookaheadDepth = 0;
		}
		
		// GOP length (keyframe interval)
		if (cfg.gop_length > 0) {
			encoder_config.gopLength = cfg.gop_length;
		}
		// else keep preset default (often NVENC_INFINITE_GOPLENGTH or fps-based)
	}
}

eznve::encoder::encoder(config cfg, CUcontext ctx, std::function<void(std::string_view)> logger) : dims(cfg.dims), fps_(cfg.fps), logger(logger) {
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

	init_params.encodeGUID = (cfg.codec == codec::h264)? NV_ENC_CODEC_H264_GUID : (cfg.codec == codec::hevc) ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_AV1_GUID;
	init_params.encodeWidth = dims.x;
	init_params.encodeHeight = dims.y;
	init_params.darWidth = dims.x;
	init_params.darHeight = dims.y;
	init_params.frameRateNum = fps_.x;
	init_params.frameRateDen = fps_.y;
	init_params.enableEncodeAsync = 0;
	init_params.enablePTD = 1;

	apply_config(session, api, cfg, init_params, encoder_config);

	int buffer_count = encoder_config.frameIntervalP + 8;

	if(encoder_config.rcParams.enableLookahead) {
		buffer_count += encoder_config.rcParams.lookaheadDepth;
	}

	//encoder_config.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 2;
	//encoder_config.profileGUID = NV_ENC_HEVC_PROFILE_MAIN10_GUID;

	CHECK_NVENC(funcs.nvEncInitializeEncoder(session, &init_params));

	for (int i = 0; i < buffer_count; i++) {
		push_buffer();
		free_buffers.push(buffers.size() - 1);
	}

	auto out_buf = pbuf_as<NV_ENC_CREATE_BITSTREAM_BUFFER>();
	out_buf->version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
	CHECK_NVENC(api.funcs().nvEncCreateBitstreamBuffer(session, out_buf));

	max_in_flight = buffer_count - 4;

	logger(std::format("encoder initialized with {} buffers", buffer_count));
}

eznve::encoder::~encoder() {
	const auto& funcs = api.funcs();

	if (frames_encoded > 0) 
		try {
			flush();
		} catch (...) {}


	for (auto& buf : buffers) {
		funcs.nvEncDestroyBitstreamBuffer(session, buf.output_stream);
		funcs.nvEncUnregisterResource(session, buf.registration);
		cuMemFree(buf.ptr);
	}

	funcs.nvEncDestroyEncoder(session);
}

std::vector<eznve::chunk> eznve::encoder::submit_frame(frame_flag flag) {

	logger(std::format("submitting frame {} ({} free, {} used)", frames_encoded, free_buffers.size(), used_buffers.size()));

	const auto& funcs = api.funcs();

	auto buffer_index = free_buffers.front();
	logger(std::format("using buffer {}", buffer_index));
	auto& buf = buffers[buffer_index];
	free_buffers.pop();
	used_buffers.push(buffer_index);

	buf.map(session);

	auto pic_params = pbuf_as<NV_ENC_PIC_PARAMS>();
	pic_params->version = NV_ENC_PIC_PARAMS_VER;
	pic_params->bufferFmt = NV_ENC_BUFFER_FORMAT_ABGR;
	pic_params->pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
	pic_params->inputBuffer = buf.mapped;
	pic_params->outputBitstream = buf.output_stream;
	pic_params->inputPitch = dims.x * 4;
	pic_params->inputTimeStamp = frames_encoded;

	if(flag == frame_flag::idr || frames_encoded == 0) {
		pic_params->encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
	}

	if(frames_encoded == 0) {
		pic_params->encodePicFlags |= NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
	}

	auto frame_status = funcs.nvEncEncodePicture(session, pic_params);
	frames_encoded++;

	std::vector<chunk> chunks;

	logger(std::format("frame status: {}", get_error(frame_status)));

	if(frame_status != NV_ENC_ERR_NEED_MORE_INPUT) {
		CHECK_NVENC(frame_status);
	} 

	if(used_buffers.size() >= max_in_flight) {

		auto& oldest_buffer = buffers[used_buffers.front()];
		auto chunk = oldest_buffer.lock(session);
		if(chunk.data.size() == 0) {
			return chunks;
		}
		logger(std::format("submitted frame {} and got {} bytes of output", frames_encoded, chunk.data.size()));

		chunks.emplace_back(std::move(chunk));
		oldest_buffer.unlock(session);
		oldest_buffer.unmap(session);
		free_buffers.push(used_buffers.front());
		used_buffers.pop();

	}

	return chunks;
}

std::vector<eznve::chunk> eznve::encoder::flush() {


	logger(std::format("processed {} frames before flushing", frames_encoded));

	bytes_encoded = 0;
	frames_encoded = 0;

	std::vector<chunk> chunks;

	auto pic_params = pbuf_as<NV_ENC_PIC_PARAMS>();
	pic_params->version = NV_ENC_PIC_PARAMS_VER;
	pic_params->encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	//pic_params->outputBitstream = out_stream;

	auto frame_status = api.funcs().nvEncEncodePicture(session, pic_params);

	CHECK_NVENC(frame_status);
	
	while(used_buffers.size() > 0) {
		auto buffer_index = used_buffers.front();
		auto& buf = buffers[buffer_index];
		auto chunk = buf.lock(session);
		chunks.emplace_back(std::move(chunk));
		buf.unlock(session);
		buf.unmap(session);
		free_buffers.push(buffer_index);
		used_buffers.pop();
	}

	return chunks;
}

void eznve::encoder::push_buffer()
{
	auto buf = buffer_t{};

	cuMemAlloc(&buf.ptr, dims.x * dims.y * 4);

	auto input_res = pbuf_as<NV_ENC_REGISTER_RESOURCE>();
	input_res->version = NV_ENC_REGISTER_RESOURCE_VER;
	input_res->resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
	input_res->width = dims.x;
	input_res->height = dims.y;
	input_res->pitch = dims.x * 4;
	input_res->subResourceIndex = 0;
	input_res->resourceToRegister = reinterpret_cast<void*>(buf.ptr); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
	input_res->bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
	input_res->bufferUsage = NV_ENC_INPUT_IMAGE;
	CHECK_NVENC(api.funcs().nvEncRegisterResource(session, input_res));
	buf.registration = input_res->registeredResource;

	auto out_buf = pbuf_as<NV_ENC_CREATE_BITSTREAM_BUFFER>();
	out_buf->version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
	CHECK_NVENC(api.funcs().nvEncCreateBitstreamBuffer(session, out_buf));
	buf.output_stream = out_buf->bitstreamBuffer;

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

eznve::chunk eznve::encoder::buffer_t::lock(void* session)
{
	NV_ENC_LOCK_BITSTREAM lock;
	std::memset(&lock, 0, sizeof(lock));
	lock.version = NV_ENC_LOCK_BITSTREAM_VER;
	lock.outputBitstream = output_stream;
	lock.doNotWait = 0;

	auto result = api.funcs().nvEncLockBitstream(session, &lock);

	CHECK_NVENC(result);

	auto chunk_span = std::span<const char>{ (const char*)lock.bitstreamBufferPtr, lock.bitstreamSizeInBytes };

	return eznve::chunk{
		.data = {chunk_span.begin(), chunk_span.end()},
		.index = lock.frameIdx,
		.timestamp = lock.outputTimeStamp,
		.duration = lock.outputDuration
	};
}

void eznve::encoder::buffer_t::unlock(void* session)
{
	CHECK_NVENC(api.funcs().nvEncUnlockBitstream(session, output_stream));
}
