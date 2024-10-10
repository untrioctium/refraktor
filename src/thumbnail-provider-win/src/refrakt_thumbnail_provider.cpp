#include <Shlwapi.h>
#include <thumbcache.h>
#include <new>
#include <chrono>
#include <cstdio>

#include <librefrakt/util/filesystem.h>
#include <librefrakt/flame_info.h>
#include <librefrakt/flame_compiler.h>

#include <librefrakt/image/converter.h>
#include <librefrakt/image/tonemapper.h>
#include <librefrakt/interface/denoiser.h>

#include <librefrakt/util/stb.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

struct shared_state {
	bool valid = false;

	rfkt::flamedb fdb;
	std::shared_ptr<ezrtc::compiler> kernel_manager;
	std::unique_ptr<rfkt::flame_compiler> flame_compiler;

	std::unique_ptr<rfkt::converter> converter;
	std::unique_ptr<rfkt::tonemapper> tonemapper;
	std::unique_ptr<rfkt::denoiser> denoiser;

	std::unique_ptr<roccu::gpu_stream> stream;
	std::unique_ptr<roccu::gpu_event> event;

	std::move_only_function<std::vector<uchar4>(std::string_view fxml, uint2 dims)> render_flame;
};

extern HINSTANCE dll_instance;

template<typename... Args>
void write_log(const char* format, Args... args)
{
	char buffer[1024];
	sprintf_s(buffer, format, args...);

	auto unix_timestamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	char buffer2[1024];
	sprintf_s(buffer2, "%d: (%d, %p) %s", unix_timestamp, GetCurrentProcessId(), dll_instance, buffer);

	// write to a log file
	FILE* f = nullptr;
	fopen_s(&f, "C:\\Users\\Public\\flam3.log", "a");
	if (f)
	{
		fprintf(f, "%s\n", buffer2);
		fclose(f);
	}
}

#define CHECK_ROCCU_THROWING(expr) do { if (auto ret = expr; ret != RU_SUCCESS) throw std::runtime_error(std::format("error executing `{}`: {}", #expr, ret)); } while (0)

auto render_flame(shared_state& state, std::string_view fxml, uint2 dims) -> std::vector<uchar4> {
	
	auto flame = rfkt::import_flam3(state.fdb, fxml);

	if(!flame) {
		SPDLOG_ERROR("Failed to import flame: {}", fxml);
		return {};
	}
	SPDLOG_INFO("Imported flame: {}", flame->name);

	auto compile_result = state.flame_compiler->get_flame_kernel(state.fdb, rfkt::precision::f32, flame.value());

	if (!compile_result.kernel) {
		SPDLOG_ERROR("Failed to compile flame: {}", compile_result.log);
		return {};
	}
	SPDLOG_INFO("Compiled flame: {}", flame->name);

	auto samples = std::vector<double>{};
	auto packer = [&samples](double v) { samples.push_back(v); };
	auto invoker = [](std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args) { return iv; };

	flame->pack_samples(packer, invoker, 0, 0, 4, dims.x, dims.y);

	auto tonemap_args = rfkt::tonemapper::args_t{};
	tonemap_args.brightness = flame->brightness.sample(0, invoker);
	tonemap_args.gamma = flame->gamma.sample(0, invoker);
	tonemap_args.vibrancy = flame->vibrancy.sample(0, invoker);

	auto fstate = compile_result.kernel->warmup(*state.stream, samples, dims, 0xDEADBEEF, 100);
	auto bin_future = compile_result.kernel->bin(*state.stream, fstate, { .millis = 500, .quality = 128 });
	state.stream->sync();
	auto bin_result = bin_future.get();

	tonemap_args.quality = bin_result.quality;
	tonemap_args.max_density = bin_result.max_density;

	auto tonemapped = roccu::gpu_image<half3>(dims, *state.stream);
	auto denoised = roccu::gpu_image<half3>(dims, *state.stream);
	auto output = roccu::gpu_image<uchar4>(dims, *state.stream);

	state.tonemapper->run(fstate.bins, tonemapped, tonemap_args, *state.stream);
	state.denoiser->denoise(tonemapped, denoised, *state.event);
	state.converter->to_uchar4(denoised, output, *state.stream, rfkt::convert_flip | rfkt::convert_swap_channels);

	state.stream->sync();

	SPDLOG_INFO("Rendered flame thumbnail for {}", flame->name);

	return output.to_host();
}

auto create_state() -> std::unique_ptr<shared_state> {

	auto state = std::make_unique<shared_state>();

	char module_buffer[MAX_PATH];
	GetModuleFileNameA(dll_instance, module_buffer, MAX_PATH);
	auto module_path = rfkt::fs::path(module_buffer).parent_path();

	write_log("Module path: %s", module_path.string().c_str());
	rfkt::fs::set_working_directory(module_path);

	auto sink = std::make_shared<spdlog::sinks::basic_file_sink_st>(std::format("C:\\Users\\Public\\flam3-thumb-{}.log", GetCurrentProcessId()), true);
	spdlog::default_logger_raw()->sinks().push_back(sink);
	spdlog::flush_on(spdlog::level::info);

	SPDLOG_INFO("Creating state");

	if (auto api = roccuInit(); api == ROCCU_API_NONE) {
		SPDLOG_ERROR("No ROCCU API available");
		return nullptr;
	}

	RUdevice dev;
	RUcontext ctx;

	CHECK_ROCCU_THROWING(ruInit(0));
	CHECK_ROCCU_THROWING(ruDeviceGet(&dev, 0));
	CHECK_ROCCU_THROWING(ruCtxCreate(&ctx, 0x01 | 0x08, dev));

	rfkt::initialize(state->fdb, "config");

	state->stream = std::make_unique<roccu::gpu_stream>();
	state->event = std::make_unique<roccu::gpu_event>();

	auto kernel = std::make_shared<ezrtc::sqlite_cache>((rfkt::fs::user_local_directory() / "kernel.sqlite3").string());
	auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(kernel);
	state->kernel_manager = std::make_shared<ezrtc::compiler>(zlib);
	state->flame_compiler = std::make_unique<rfkt::flame_compiler>(state->kernel_manager);

	state->tonemapper = std::make_unique<rfkt::tonemapper>(*state->kernel_manager);
	state->converter = std::make_unique<rfkt::converter>(*state->kernel_manager);
	state->denoiser = rfkt::denoiser::make("rfkt::oidn_denoiser", { 1024, 1024 }, rfkt::denoiser_flag::none, *state->stream);

	state->render_flame = [&state = *state](std::string_view fxml, uint2 dims) -> std::vector<uchar4> {
		try {
			return render_flame(state, fxml, dims);
		}
		catch (const std::exception& e) {
			SPDLOG_ERROR("Error rendering flame: {}", e.what());
			return {};
		}
	};

	SPDLOG_INFO("State created");
	return state;

}

auto get_shared_state() {
	static std::unique_ptr<shared_state> ss = []() -> std::unique_ptr<shared_state> {
		try { return create_state(); }
		catch (const std::exception& e) {
			write_log("Error creating shared state: %s", e.what());
			return nullptr;
		}
	}();
	return ss.get();
}



class Flam3ThumbProvider : public IInitializeWithStream, public IThumbnailProvider
{
public:
	Flam3ThumbProvider() : ref_count(1) { (void)get_shared_state(); }
	~Flam3ThumbProvider() {}

	// IUnknown
	IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
	{
		static const QITAB qit[] = {
			QITABENT(Flam3ThumbProvider, IInitializeWithStream),
			QITABENT(Flam3ThumbProvider, IThumbnailProvider),
			{ 0 },
		};
		return QISearch(this, qit, riid, ppv);
	}

	IFACEMETHODIMP_(ULONG) AddRef()
	{
		return InterlockedIncrement(&ref_count);
	}

	IFACEMETHODIMP_(ULONG) Release()
	{
		ULONG ref = InterlockedDecrement(&ref_count);
		if (ref == 0) delete this;
		return ref;
	}

	// IInitializeWithStream
	IFACEMETHODIMP Initialize(IStream* stream, DWORD grf_mode);

	// IThumbnailProvider
	IFACEMETHODIMP GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha);

private:

	long ref_count;
	std::string xml_data;

};

HRESULT Flam3ThumbProvider_CreateInstance(REFIID riid, void** ppv) {
	Flam3ThumbProvider* provider = new (std::nothrow) Flam3ThumbProvider();
	if (!provider) return E_OUTOFMEMORY;
	
	auto hr = provider->QueryInterface(riid, ppv);
	provider->Release();
	return hr;
}

HRESULT Flam3ThumbProvider::Initialize(IStream* stream, DWORD grf_mode) {
	
	char buffer[1024];
	ULONG read = 0;
	while(SUCCEEDED(stream->Read(buffer, sizeof(buffer), &read)) && read > 0) {
		xml_data.append(buffer, read);
	}

	return S_OK;
}

HRESULT Flam3ThumbProvider::GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha) {
	SPDLOG_INFO("Flam3ThumbProvider::GetThumbnail({})", cx);

	auto ss = get_shared_state();
	if (!ss) {
		SPDLOG_ERROR("Failed to get shared state");
		return E_UNEXPECTED;
	}

	auto dims = uint2{ 1280, 720 };

	auto ret = get_shared_state()->render_flame(xml_data, { dims.x, dims.y });

	if (ret.empty()) {
		SPDLOG_INFO("Failed to render flame");
		return E_UNEXPECTED;
	}

	// create a bitmap from the image data
	BITMAPINFOHEADER bi = { sizeof(BITMAPINFOHEADER), dims.x, dims.y, 1, 32, BI_RGB };
	*phbmp = CreateDIBitmap(GetDC(nullptr), &bi, CBM_INIT, ret.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS);

	if(!*phbmp) {
		SPDLOG_ERROR("Failed to create bitmap");
		return E_UNEXPECTED;
	}

	//auto xml_hash = rfkt::hash::calc(xml_data);
	//auto out_path = std::format("C:\\Users\\Public\\flam3-thumb-{}-{}.png", GetCurrentProcessId(), xml_hash.str32());

	//rfkt::stbi::write_file(ret.data(), cx, cx, out_path);

	SPDLOG_INFO("Thumbnail created");

	*pdwAlpha = WTSAT_ARGB;
	return S_OK;
}