#include <imftw/imftw.h>

#include "signals_internal.h"
#include "events.h"

#include <imgui_impl_opengl3.h>

#ifdef _WIN32
#include <commdlg.h>
#endif

namespace imftw {
	context_t& context() {
		static context_t ctx;
		return ctx;
	}
}

void setup_event_callbacks(imftw::context_t& ctx) {
	auto window = ctx.window;

	glfwSetWindowSizeCallback(window, [](GLFWwindow*, int width, int height) {
		if (width == 0 || height == 0) return;
		imftw::context().window_size.store({ width, height });
	});

	glfwSetWindowPosCallback(window, [](GLFWwindow*, int x, int y) {
		imftw::context().window_position.store({ x, y });
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
		imftw::push_event(imftw::events::key{ key, scancode, action, get_key_modifiers(window), glfwGetKeyName(key, scancode) });
	});

	glfwSetCursorPosCallback(window, [](GLFWwindow*, double x, double y) {
		imftw::push_event(imftw::events::mouse_move{ x, y });
	});

	glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int) {
		imftw::push_event(imftw::events::mouse_button{ button, action, get_key_modifiers(window)});
	});

	glfwSetWindowCloseCallback(window, [](GLFWwindow*) {
		//imftw::push_event(imftw::events::window_close{});
	});

	glfwSetCharCallback(window, [](GLFWwindow*, unsigned int codepoint) {
		imftw::push_event(imftw::events::char_input{ codepoint });
	});

	glfwSetScrollCallback(window, [](GLFWwindow*, double x, double y) {
		imftw::push_event(imftw::events::mouse_scroll{ x, y });
	});

	glfwSetWindowIconifyCallback(window, [](GLFWwindow*, int iconified) {
		imftw::context().iconified.store(iconified == GLFW_TRUE);
	});

	glfwSetWindowMaximizeCallback(window, [](GLFWwindow*, int maximized) {
		imftw::context().maximized.store(maximized == GLFW_TRUE);
	});
}

void setup_imgui(imftw::context_t& ctx) {
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
	io.Fonts->AddFontFromFileTTF("assets/fonts/Roboto-Regular.ttf", 14);
	io.Fonts->AddFontFromFileTTF("assets/fonts/OpenSans-Medium.ttf", 16);
	io.Fonts->AddFontFromFileTTF("assets/fonts/Montserrat-Thin.ttf", 14);

	ImGui_ImplOpenGL3_Init("#version 460");
}

void imftw::run(std::string_view window_title, std::move_only_function<void()>&& main_function) {

	auto& ctx = context();

	if (!glfwInit()) {
		return;
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
	auto decorated_main = [func = std::move(main_function), stopper = std::move(stopper)]() mutable {
		glfwMakeContextCurrent(context().window);

		func();

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
			auto sig = sig::poll_glfw_signal();
			if (!sig) break;

			std::visit([&ctx](auto&& sig) { sig.handle(ctx); }, std::move(*sig));
		}
	}

	main_thread.join();

#ifdef _WIN32
	ctx.taskbar->Release();
	CoUninitialize();
#endif
}

void imftw::begin_frame(ImVec4 clear_color) {
	auto& ctx = context();

	assert(ctx.opengl_thread_id == std::this_thread::get_id());

	ctx.frame_start = context_t::clock_t::now();

	while (true) {
		auto sig = imftw::sig::poll_opengl_signal();
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
		imftw::sig::set_cursor(ImGui::GetMouseCursor());
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

void imftw::end_frame(bool render) {

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

#ifdef _WIN32
template<decltype(&GetOpenFileNameA) func, decltype(OPENFILENAMEA::Flags) flags = 0>
std::string dialog_impl(std::string_view filter, std::string path)
{
	OPENFILENAMEA open;
	memset(&open, 0, sizeof(open));

	char filename[512];
	memset(&filename, 0, sizeof(filename));

	open.lStructSize = sizeof(open);
	open.hwndOwner = imftw::context().hwnd;
	open.lpstrFile = filename;
	open.lpstrInitialDir = path.c_str();
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

std::string imftw::show_open_dialog(const std::filesystem::path& path, std::string_view filter)
{
#ifdef _WIN32
	return dialog_impl<GetOpenFileNameA, OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST>(filter, path.string());
#endif
}

std::string imftw::show_save_dialog(const std::filesystem::path& path, std::string_view filter)
{
#ifdef _WIN32
	return dialog_impl<GetSaveFileNameA, OFN_OVERWRITEPROMPT>(filter, path.string());
#endif
}