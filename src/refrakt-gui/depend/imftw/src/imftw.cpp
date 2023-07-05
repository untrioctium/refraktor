#include <imftw/imftw.h>

#include "signals_internal.h"
#include "events.h"

#include <imgui_impl_opengl3.h>
#include <imgui_freetype.h>

#ifdef _WIN32
#include <commdlg.h>
#include <shellapi.h>
#endif

namespace ImFtw {
	context_t& context() {
		static context_t ctx;
		return ctx;
	}
}

void setup_event_callbacks(ImFtw::context_t& ctx) {
	auto window = ctx.window;

	glfwSetWindowSizeCallback(window, [](GLFWwindow*, int width, int height) {
		if (width == 0 || height == 0) return;
		ImFtw::context().window_size.store({ width, height });
	});

	glfwSetWindowPosCallback(window, [](GLFWwindow*, int x, int y) {
		ImFtw::context().window_position.store({ x, y });
	});

	constexpr static auto get_key_modifiers = [](GLFWwindow* window) {
		int mods = 0;
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS)
			mods |= GLFW_MOD_CONTROL;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
			mods |= GLFW_MOD_SHIFT;
		if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS)
			mods |= GLFW_MOD_ALT;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS)
			mods |= GLFW_MOD_SUPER;
		return mods;
	};

	glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int) {
		ImFtw::push_event(ImFtw::events::key{ key, scancode, action, get_key_modifiers(window), glfwGetKeyName(key, scancode) });
	});

	glfwSetCursorPosCallback(window, [](GLFWwindow*, double x, double y) {
		ImFtw::push_event(ImFtw::events::mouse_move{ x, y });
	});

	glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int) {
		ImFtw::push_event(ImFtw::events::mouse_button{ button, action, get_key_modifiers(window)});
	});

	glfwSetWindowCloseCallback(window, [](GLFWwindow*) {
		//ImFtw::push_event(ImFtw::events::window_close{});
	});

	glfwSetCharCallback(window, [](GLFWwindow*, unsigned int codepoint) {
		ImFtw::push_event(ImFtw::events::char_input{ codepoint });
	});

	glfwSetScrollCallback(window, [](GLFWwindow*, double x, double y) {
		ImFtw::push_event(ImFtw::events::mouse_scroll{ x, y });
	});

	glfwSetWindowIconifyCallback(window, [](GLFWwindow*, int iconified) {
		ImFtw::context().iconified.store(iconified == GLFW_TRUE);
	});

	glfwSetWindowMaximizeCallback(window, [](GLFWwindow*, int maximized) {
		ImFtw::context().maximized.store(maximized == GLFW_TRUE);
	});
}

void setup_imgui(ImFtw::context_t& ctx) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableSetMousePos;
	io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors | ImGuiBackendFlags_HasSetMousePos;

	ctx.cursors[ImGuiMouseCursor_Arrow] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	ctx.cursors[ImGuiMouseCursor_TextInput] = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
	ctx.cursors[ImGuiMouseCursor_ResizeNS] = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
	ctx.cursors[ImGuiMouseCursor_ResizeEW] = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
	ctx.cursors[ImGuiMouseCursor_Hand] = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
	ctx.cursors[ImGuiMouseCursor_ResizeAll] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	ctx.cursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	ctx.cursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	ctx.cursors[ImGuiMouseCursor_NotAllowed] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

	io.ConfigWindowsMoveFromTitleBarOnly = true;
	float font_size = 17.0f;
	ImFontGlyphRangesBuilder builder;
	builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
	builder.AddChar(0x03BB);
	builder.AddChar(0x00BA);
	ImVector<ImWchar> ranges;
	builder.BuildRanges(&ranges);

	ImFontConfig config;
	config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;

	io.Fonts->AddFontFromFileTTF("C:/windows/fonts/segoeui.ttf", font_size, &config, ranges.Data);

	static const ImWchar icons_ranges[] = { 0xe000, 0x10fffd, 0 };
	ImFontConfig icons_config;
	icons_config.MergeMode = true;
	icons_config.PixelSnapH = true;
	icons_config.GlyphMinAdvanceX = font_size;
	icons_config.GlyphMaxAdvanceX = font_size;
	icons_config.GlyphOffset.y = 4.0f;
	io.Fonts->AddFontFromFileTTF("assets/fonts/MaterialIcons-Regular.ttf", icons_config.GlyphMinAdvanceX, &icons_config, icons_ranges);
	io.Fonts->Build();

	ImGui_ImplOpenGL3_Init("#version 460");
}

int ImFtw::Run(std::string_view window_title, std::move_only_function<int()>&& main_function) {

	auto& ctx = context();

	if (!glfwInit()) {
		return 1;
	}
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

	ctx.monitor = glfwGetPrimaryMonitor();
	ctx.window = glfwCreateWindow(1920, 1080, window_title.data(), nullptr, nullptr);

#ifdef _WIN32
	ctx.hwnd = glfwGetWin32Window(ctx.window);
	BOOL dark_value = TRUE;
	auto window_backdrop_value = DWM_SYSTEMBACKDROP_TYPE::DWMSBT_NONE;
	DwmSetWindowAttribute(ctx.hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark_value, sizeof(dark_value));
	DwmSetWindowAttribute(ctx.hwnd, DWMWA_SYSTEMBACKDROP_TYPE, &window_backdrop_value, sizeof(window_backdrop_value));

	CoInitialize(nullptr);
	CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER, IID_ITaskbarList4, (void**)&ctx.taskbar);
#endif

	setup_event_callbacks(ctx);
	glfwSetInputMode(ctx.window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

	glfwMakeContextCurrent(ctx.window);
	gladLoadGL();

	setup_imgui(ctx);
	glfwMakeContextCurrent(nullptr);

	ctx.last_time = glfwGetTime();

	std::stop_source stopper;
	auto stop_token = stopper.get_token();

	auto promise = std::promise<int>{};
	auto future = promise.get_future();

	auto decorated_main = [func = std::move(main_function), stopper = std::move(stopper), promise = std::move(promise)]() mutable {
		glfwMakeContextCurrent(context().window);

		int ret = 0;
		try {
			promise.set_value_at_thread_exit(func());
		} catch (...) {
			promise.set_exception_at_thread_exit(std::current_exception());
		}

		stopper.request_stop();
		glfwPostEmptyEvent();
	};

	std::jthread main_thread(std::move(decorated_main));
	ctx.opengl_thread_id = main_thread.get_id();
	
	while (!stop_token.stop_requested()) {
		// wait for at least one input event
		// these will actually be processed on the rendering thread
		glfwWaitEvents();

		// poll any glfw signals, which are commands that can
		// only be done on the main thread
		while (true) {
			auto sig = Sig::poll_glfw_signal();
			if (!sig) break;

			std::visit([&ctx](auto sig) { sig.handle(ctx); }, std::move(*sig));
		}
	}

	main_thread.join();

#ifdef _WIN32
	ctx.taskbar->Release();
	CoUninitialize();
#endif

	return future.get();
}

void ImFtw::OpenBrowser(std::string_view url) {
	#ifdef _WIN32
	ShellExecuteA(nullptr, "open", url.data(), nullptr, nullptr, SW_SHOWNORMAL);
	#endif
}

void ImFtw::BeginFrame(ImVec4 clear_color) {
	auto& ctx = context();

	assert(ctx.opengl_thread_id == std::this_thread::get_id());

	ctx.frame_start = context_t::clock_t::now();

	while (true) {
		auto sig = ImFtw::Sig::poll_opengl_signal();
		if(!sig) break;

		std::visit([&ctx](auto&& sig) { sig.handle(ctx); }, std::move(*sig));
	}

	if (ctx.target_framerate) {
		ctx.frame_end_target = ctx.frame_start + std::chrono::nanoseconds(1'000'000'000 / *ctx.target_framerate);
	}

	glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	auto& io = ImGui::GetIO();
	auto window_size = ctx.window_size.load();
	io.DisplaySize = ImVec2((float)window_size.first, (float)window_size.second);

	auto time = glfwGetTime();
	io.DeltaTime = static_cast<decltype(io.DeltaTime)>(time - ctx.last_time);
	ctx.last_time = time;

	if (ctx.cursor_enabled) {
		ImFtw::Sig::SetCursor(ImGui::GetMouseCursor());
	}

	while (true) {
		auto event = poll_event();
		if (!event) break;

		std::visit([&io](auto&& ev) {
			ev.handle(io);
		}, std::move(event.value()));
	}

	ImGui_ImplOpenGL3_NewFrame();
	ImGui::NewFrame();

	while (true) {
		auto func = std::move_only_function<void()>{};
		if (!ctx.deferred_functions.try_dequeue(func)) break;
		func();
	}
}

void precise_sleep(double seconds) {
	using namespace std;
	using namespace std::chrono;

	static double estimate = 5e-3;
	static double mean = 5e-3;
	static double m2 = 0;
	static int64_t count = 1;

	while (seconds > estimate) {
		auto start = high_resolution_clock::now();
		this_thread::sleep_for(milliseconds(1));
		auto end = high_resolution_clock::now();

		double observed = (end - start).count() / 1e9;
		seconds -= observed;

		++count;
		double delta = observed - mean;
		mean += delta / count;
		m2 += delta * (observed - mean);
		double stddev = sqrt(m2 / (count - 1));
		estimate = mean + stddev;
	}

	// spin lock
	auto start = high_resolution_clock::now();
	while ((high_resolution_clock::now() - start).count() / 1e9 < seconds);
}

void ImFtw::EndFrame(bool render) {

	auto& ctx = context();

	assert(ctx.opengl_thread_id == std::this_thread::get_id());

	ImGui::Render();
	if (render && !ctx.iconified) {
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(ctx.window);
	}

	if (ctx.target_framerate) {
		precise_sleep(std::chrono::duration_cast<std::chrono::nanoseconds>(ctx.frame_end_target - context_t::clock_t::now()).count() / 1e9);
	}
}

void ImFtw::DeferNextFrame(std::move_only_function<void()>&& func)
{
	context().deferred_functions.enqueue(std::move(func));
}

bool ImFtw::OnRenderingThread()
{
	thread_local auto thread_id = std::this_thread::get_id();
	return context().opengl_thread_id == thread_id;
}

#ifdef _WIN32
template<decltype(&GetOpenFileNameA) func, decltype(OPENFILENAMEA::Flags) flags = 0>
std::string dialog_impl(std::string_view filter, std::string_view path)
{
	OPENFILENAMEA open;
	memset(&open, 0, sizeof(open));

	char filename[512];
	memset(&filename, 0, sizeof(filename));

	open.lStructSize = sizeof(open);
	open.hwndOwner = ImFtw::context().hwnd;
	open.lpstrFile = filename;
	open.lpstrInitialDir = path.data();
	open.nMaxFile = sizeof(filename);

	if (!filter.empty()) {
		open.lpstrFilter = filter.data();
		open.nFilterIndex = 1;
	}

	open.Flags = flags | OFN_NOCHANGEDIR;

	if ((*func)(&open) == TRUE) return open.lpstrFile;
	return {};
}
#endif

std::string ImFtw::ShowOpenDialog(const std::filesystem::path& path, std::string_view filter)
{
#ifdef _WIN32
	return dialog_impl<&GetOpenFileNameA, OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST>(filter, path.string());
#endif
}

std::string ImFtw::ShowSaveDialog(const std::filesystem::path& path, std::string_view filter)
{
#ifdef _WIN32
	return dialog_impl<&GetSaveFileNameA, OFN_OVERWRITEPROMPT>(filter, path.string());
#endif
}

double ImFtw::Time() {
	return glfwGetTime();
}