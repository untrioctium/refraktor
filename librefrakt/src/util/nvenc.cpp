#include <ffnvcodec/nvEncodeAPI.h>

#include <librefrakt/util/nvenc.h>
#include <librefrakt/util/platform.h>

#define NVENC_SAFE_CALL(x) \
	do{ \
		auto nvsaferet = x; \
		if(nvsaferet != NV_ENC_SUCCESS) { \
			fmt::print("NVENC '{}' failed with: {}\n", #x, last_error()); \
			exit(1); \
		} \
	} while (0) \

namespace rfkt::nvenc::detail {
	class api_t {
	public:
		api_t() {
			std::string libname = (platform::is_posix()) ? "libnvidia-encode.so" : "nvEncodeAPI64";

			lib = platform::dynlib::load(libname);
			auto create_instance = lib->symbol<nv_create_api>("NvEncodeAPICreateInstance");
			auto check_version = lib->symbol<nv_check_version>("NvEncodeAPIGetMaxSupportedVersion");

			uint32_t d_ver = 0;
			uint32_t h_ver = (NVENCAPI_MAJOR_VERSION << 4) | NVENCAPI_MINOR_VERSION;
			check_version(&d_ver);
			if (h_ver > d_ver) throw 0;

			funcs_ = new NV_ENCODE_API_FUNCTION_LIST();
			funcs_->version = NV_ENCODE_API_FUNCTION_LIST_VER;
			auto ret = create_instance(funcs_);
			if (ret != NV_ENC_SUCCESS) throw 0;
		}

		auto funcs() const { return *funcs_; }

		auto open_session(void** sesh) -> NVENCSTATUS const {
			NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sesh_p = {
				NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
				NV_ENC_DEVICE_TYPE_CUDA,
				(CUcontext)cuda::context::current(),
				0,
				NVENCAPI_VERSION,
				0,
				0
			};

			auto status = funcs_->nvEncOpenEncodeSessionEx(&sesh_p, sesh);

			return status;
		}

	private:
		using nv_create_api = NVENCSTATUS(NVENCAPI*)(NV_ENCODE_API_FUNCTION_LIST*);
		using nv_check_version = NVENCSTATUS(NVENCAPI*)(uint32_t*);

		std::unique_ptr<platform::dynlib> lib;
		NV_ENCODE_API_FUNCTION_LIST* funcs_;
	};

	detail::api_t& api() {
		static auto g_api = detail::api_t();
		return g_api;
	}

	const std::string& get_error(NVENCSTATUS code) {
		static const std::map<NVENCSTATUS, std::string> codes{
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

	const GUID& get_codec_guid(codec c) {
		static const std::map<codec, GUID> codecs{
			{codec::h264, NV_ENC_CODEC_H264_GUID},
			{codec::hevc, NV_ENC_CODEC_HEVC_GUID}
		};

		return codecs.at(c);
	}

	NV_ENC_BUFFER_FORMAT get_buffer_format(buffer_format bf) {
		static const std::map<buffer_format, NV_ENC_BUFFER_FORMAT> buffer_formats{
			{buffer_format::undefined, NV_ENC_BUFFER_FORMAT_UNDEFINED},
			{buffer_format::nv12, NV_ENC_BUFFER_FORMAT_NV12},
			{buffer_format::yv12, NV_ENC_BUFFER_FORMAT_YV12},
			{buffer_format::iyuv, NV_ENC_BUFFER_FORMAT_IYUV},
			{buffer_format::yuv444, NV_ENC_BUFFER_FORMAT_YUV444},
			{buffer_format::yuv420_10bit, NV_ENC_BUFFER_FORMAT_YUV420_10BIT},
			{buffer_format::yuv444_10bit, NV_ENC_BUFFER_FORMAT_YUV444_10BIT},
			{buffer_format::argb, NV_ENC_BUFFER_FORMAT_ARGB},
			{buffer_format::argb10, NV_ENC_BUFFER_FORMAT_ARGB10},
			{buffer_format::ayuv, NV_ENC_BUFFER_FORMAT_AYUV},
			{buffer_format::abgr, NV_ENC_BUFFER_FORMAT_ABGR},
			{buffer_format::abgr10, NV_ENC_BUFFER_FORMAT_ABGR10},
			{buffer_format::u8, NV_ENC_BUFFER_FORMAT_U8}
		};

		return buffer_formats.at(bf);
	}
}


auto rfkt::nvenc::session::make() -> std::unique_ptr<session> {
	std::unique_ptr<session> sesh{ new session{} };

	auto ret = detail::api().open_session(&sesh->sesh);
	if (ret != NV_ENC_SUCCESS) throw ret;
	return sesh;
}

std::string rfkt::nvenc::session::last_error() {
	return detail::api().funcs().nvEncGetLastErrorString(sesh);
}

auto rfkt::nvenc::session::initialize(std::pair<uint32_t, uint32_t> dims, std::pair<uint32_t, uint32_t> fps) -> std::shared_ptr<cuda_buffer<uchar4>> {
	const auto& funcs = detail::api().funcs();

	dims_ = dims;
	NV_ENC_INITIALIZE_PARAMS init_params{};
	init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
	init_params.encodeGUID = detail::get_codec_guid(codec::h264);
	init_params.presetGUID = NV_ENC_PRESET_HQ_GUID;
	init_params.encodeWidth = dims.first;
	init_params.encodeHeight = dims.second;
	init_params.darWidth = dims.first;
	init_params.darHeight = dims.second;
	init_params.frameRateNum = fps.first;
	init_params.frameRateDen = fps.second;
	init_params.enableEncodeAsync = 0;
	init_params.enablePTD = 1;

	NV_ENC_PRESET_CONFIG preset_config;
	memset(&preset_config, 0, sizeof(NV_ENC_PRESET_CONFIG));
	preset_config.version = NV_ENC_PRESET_CONFIG_VER;
	preset_config.presetCfg.version = NV_ENC_CONFIG_VER;
	NVENC_SAFE_CALL(funcs.nvEncGetEncodePresetConfig(sesh, init_params.encodeGUID, init_params.presetGUID, &preset_config));
	init_params.encodeConfig = &preset_config.presetCfg;

	NVENC_SAFE_CALL(funcs.nvEncInitializeEncoder(sesh, &init_params));

	NV_ENC_CREATE_BITSTREAM_BUFFER out_buf = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
	NVENC_SAFE_CALL(funcs.nvEncCreateBitstreamBuffer(sesh, &out_buf) != NV_ENC_SUCCESS);
	out_stream = out_buf.bitstreamBuffer;

	input_buffer = std::make_shared<cuda_buffer<uchar4>>(dims.first * dims.second);
	NV_ENC_REGISTER_RESOURCE input_res{};
	memset(&input_res, 0, sizeof(NV_ENC_REGISTER_RESOURCE));
	input_res.version = NV_ENC_REGISTER_RESOURCE_VER;
	input_res.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
	input_res.width = dims.first;
	input_res.height = dims.second;
	input_res.pitch = dims.first * 4;
	input_res.subResourceIndex = 0;
	input_res.resourceToRegister = (void*)input_buffer->ptr();
	input_res.bufferFormat = detail::get_buffer_format(buffer_format::abgr);
	input_res.bufferUsage = NV_ENC_INPUT_IMAGE;
	NVENC_SAFE_CALL(detail::api().funcs().nvEncRegisterResource(sesh, &input_res));
	in_reg = input_res.registeredResource;

	//SPDLOG_INFO("Started encode session {} with parameters {}x{}x({}/{})", sesh, dims.first, dims.second, fps.first, fps.second);

	return input_buffer;
}

std::optional<std::vector<std::byte>> rfkt::nvenc::session::submit_frame(bool idr, bool done) {
	const auto& funcs = detail::api().funcs();

	auto ret = std::optional<std::vector<std::byte>>{};

	NV_ENC_MAP_INPUT_RESOURCE in_map{};
	memset(&in_map, 0, sizeof(in_map));
	in_map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
	in_map.registeredResource = in_reg;
	NVENC_SAFE_CALL(funcs.nvEncMapInputResource(sesh, &in_map));

	NV_ENC_PIC_PARAMS pic_params{};
	memset(&pic_params, 0, sizeof(pic_params));
	pic_params.version = NV_ENC_PIC_PARAMS_VER;
	if (!done) {
		pic_params.bufferFmt = detail::get_buffer_format(buffer_format::abgr);
		pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		pic_params.inputBuffer = in_map.mappedResource;
		if (idr) pic_params.encodePicFlags = NV_ENC_PIC_FLAG_FORCEIDR;
	}
	else {
		pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	}

	auto frame_status = funcs.nvEncEncodePicture(sesh, &pic_params);

	if (frame_status != NV_ENC_ERR_NEED_MORE_INPUT) {
		NVENC_SAFE_CALL(frame_status);

		std::vector<std::byte> ret_data;

		NV_ENC_LOCK_BITSTREAM out_bitstream = { NV_ENC_LOCK_BITSTREAM_VER }; out_bitstream.outputBitstream = out_stream;
		NVENC_SAFE_CALL(funcs.nvEncLockBitstream(sesh, &out_bitstream));
		ret_data.resize(out_bitstream.bitstreamSizeInBytes);
		memcpy_s(ret_data.data(), ret_data.size(), out_bitstream.bitstreamBufferPtr, out_bitstream.bitstreamSizeInBytes);
		NVENC_SAFE_CALL(funcs.nvEncUnlockBitstream(sesh, out_bitstream.outputBitstream));

		ret = { ret_data };


		//SPDLOG_INFO("Encode session {} returning {} bytes", sesh, ret_data.size());
	}

	NVENC_SAFE_CALL(funcs.nvEncUnmapInputResource(sesh, in_map.mappedResource));

	return ret;
}

rfkt::nvenc::session::~session() {
	const auto& funcs = detail::api().funcs();

	funcs.nvEncDestroyBitstreamBuffer(sesh, out_stream);
	funcs.nvEncUnregisterResource(sesh, in_reg);
	funcs.nvEncDestroyEncoder(sesh);
}
