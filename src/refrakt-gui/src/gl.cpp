#include <librefrakt/util/cuda.h>

#include "gui.h"
#include "gl.h"

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <commdlg.h>
#endif 
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>


#include <cudaGL.h>

#include <readerwriterqueue.h>

#include <dwmapi.h>

ImGuiKey glfw_to_imgui_key(int key);

namespace rfkt::gl {
	struct state {
		using clock_t = std::chrono::steady_clock;
		using timepoint_t = decltype(std::chrono::steady_clock::now());

		GLFWwindow* window;
		GLFWmonitor* monitor;
		std::optional<unsigned int> target_fps;
		timepoint_t frame_start;
		timepoint_t frame_end;

		std::unordered_map<ImGuiMouseCursor, GLFWcursor*> cursors;
		bool cursor_enabled = true;

		std::atomic<bool> minimized = false;

	} gl_state;
}

std::string rfkt::gl::show_open_dialog(std::string_view filter)
{
#ifdef _WIN32
	OPENFILENAMEA open;
	memset(&open, 0, sizeof(open));

	char filename[512];
	memset(&filename, 0, sizeof(filename));

	std::string path = (std::filesystem::current_path() / "assets" / "flames").string();

	open.lStructSize = sizeof(open);
	open.hwndOwner = glfwGetWin32Window(gl_state.window);
	open.lpstrFile = filename;
	open.lpstrInitialDir = path.c_str();
	open.nMaxFile = sizeof(filename);

	if (!filter.empty()) {
		open.lpstrFilter = filter.data();
		open.nFilterIndex = 1;
	}

	open.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

	if (GetOpenFileNameA(&open) == TRUE) return open.lpstrFile;
	if(auto err = CommDlgExtendedError(); err != 0) SPDLOG_ERROR("Dialog error: 0x{:04x}", err);
	return {};
#endif
}

namespace rfkt {
	namespace events {
		struct mouse_move {
			double x;
			double y;
		};

		struct mouse_button {
			int button;
			int action;
			int mods;
		};

		struct key {
			int key;
			int scancode;
			int action;
			int mods;
			const char* key_name;
		};

		struct char_input {
			unsigned int codepoint;
		};

		struct scroll {
			double xoffset;
			double yoffset;
		};

		struct window_size {
			int width;
			int height;
		};

		struct window_close {
		};

		struct window_focus {
			int focused;
		};

		struct window_iconify {
			int iconified;
		};

		struct window_pos {
			int x;
			int y;
		};

		struct window_refresh {
		};

		struct window_maximize {
			int maximized;
		};

		struct framebuffer_size {
			int width;
			int height;
		};

		struct file_drop {
			int count;
			const char** paths;
		};

		struct joystick {
			int jid;
			int event;
		};

		struct clipboard {
			const char* string;
		};

		struct unknown {
			int event;
		};

		struct destroy_cuda_map {
			CUgraphicsResource cuda_res;
		};

		struct destroy_texture {
			CUgraphicsResource cuda_res;
			GLuint tex_id;
		};
	}

	using event = std::variant<
		events::mouse_move,
		events::mouse_button,
		events::key,
		events::char_input,
		events::scroll,
		events::window_size,
		events::window_close,
		events::window_focus,
		events::window_iconify,
		events::window_pos,
		events::window_refresh,
		events::window_maximize,
		events::framebuffer_size,
		events::file_drop,
		events::joystick,
		events::clipboard,
		events::destroy_cuda_map,
		events::destroy_texture,
		events::unknown>;

	std::optional<event> poll_event();

	namespace signals {
		struct set_cursor {
			GLFWcursor* cursor;
		};

		struct set_clipboard {
			std::string contents;
		};

		struct get_clipboard {
			std::promise<std::string> promise;
		};

		struct set_mouse_position {
			double x;
			double y;
		};

		struct set_cursor_enabled {
			bool enabled;
		};
	}
	using signal = std::variant<
		signals::set_cursor,
		signals::set_mouse_position,
		signals::set_cursor_enabled,
		signals::set_clipboard,
		signals::get_clipboard>;

	using signal_queue_t = moodycamel::ReaderWriterQueue<signal>;

	signal_queue_t& signal_queue() {
		static signal_queue_t queue;
		return queue;
	}

	void push_signal(signal&& sig) {
		signal_queue().enqueue(std::move(sig));
		glfwPostEmptyEvent();
	}

	std::optional<signal> poll_signal() {
		if (auto sig = signal{}; signal_queue().try_dequeue(sig)) {
			return sig;
		}
		return std::nullopt;
	}
}

void rfkt::gl::set_target_fps(unsigned int fps) {
	if (fps == 0) {
		gl_state.target_fps = std::nullopt;
	}
	else {
		gl_state.target_fps = fps;
	}
}

std::atomic<int2>& window_size_atomic() {
	static std::atomic<int2> window_size;
	return window_size;
}

std::atomic<int2>& window_pos_atomic() {
	static std::atomic<int2> window_pos;
	return window_pos;
}

int2 rfkt::gl::get_window_size() {
	return window_size_atomic().load();
}

void rfkt::gl::make_current() {
	glfwMakeContextCurrent(gl_state.window);
}

using event_queue_t = moodycamel::ReaderWriterQueue<rfkt::event>;

event_queue_t& event_queue() {
	static event_queue_t queue;
	return queue;
}

std::optional<rfkt::event> rfkt::poll_event() {
	rfkt::event event;
	if (event_queue().try_dequeue(event)) {
		return event;
	}
	return std::nullopt;
}

void push_event(rfkt::event&& event) {
	event_queue().enqueue(std::move(event));
}

rfkt::gl::texture::texture(std::size_t w, std::size_t h) : width_(w), height_(h)
{
	glGenTextures(1, &tex_id);
	glBindTexture(GL_TEXTURE_2D, tex_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	CUDA_SAFE_CALL(cuGraphicsGLRegisterImage(&cuda_res, tex_id, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
}

rfkt::gl::texture::~texture()
{
	if (tex_id != 0) {
		CUDA_SAFE_CALL(cuGraphicsUnregisterResource(cuda_res));
		glDeleteTextures(1, &tex_id);
	}
}

rfkt::gl::texture::cuda_map::cuda_map(texture& tex) : cuda_res(tex.cuda_res)
{
	CUDA_SAFE_CALL(cuGraphicsMapResources(1, &cuda_res, 0));
	CUDA_SAFE_CALL(cuGraphicsSubResourceGetMappedArray(&arr, cuda_res, 0, 0));

	memset(&copy_params, 0, sizeof(copy_params));
	copy_params.srcXInBytes = 0;
	copy_params.srcY = 0;
	copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copy_params.srcPitch = tex.width() * 4;
	copy_params.srcDevice = 0;

	copy_params.dstXInBytes = 0;
	copy_params.dstY = 0;
	copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy_params.dstArray = arr;

	copy_params.WidthInBytes = tex.width() * 4;
	copy_params.Height = tex.height();

}

rfkt::gl::texture::cuda_map::~cuda_map()
{
	if (cuda_res == nullptr) return;
	CUDA_SAFE_CALL(cuGraphicsUnmapResources(1, &cuda_res, 0));
	//push_event(rfkt::events::destroy_cuda_map{ cuda_res });
}

bool rfkt::gl::init(int width, int height)
{
	glfwSetErrorCallback([](int error, const char* description) {
		SPDLOG_ERROR("GLFW Error {}: {}\n", error, description);
		});

	if (!glfwInit()) {
		SPDLOG_CRITICAL("Couldn't initialize GLFW.\n");
		return false;
	}

	int2 window_size = { width, height };

	const char* glsl_version = "#version 460";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
	glfwWindowHint(GLFW_VISIBLE, FALSE);
	glfwWindowHint(GLFW_MAXIMIZED, TRUE);
	//glfwWindowHint(GLFW_DECORATED, 0);

	gl_state.monitor = glfwGetPrimaryMonitor();
	gl_state.window = glfwCreateWindow(window_size.x, window_size.y, "Refrakt", nullptr, nullptr);

	auto w32_handle = glfwGetWin32Window(gl_state.window);
	BOOL dark_value = TRUE;
	::DwmSetWindowAttribute(w32_handle, 20, &dark_value, sizeof(dark_value));
	glfwShowWindow(gl_state.window);

	if (gl_state.window == nullptr) {
		SPDLOG_CRITICAL("Could not create window");
		return false;
	}

	glfwGetWindowSize(gl_state.window, &window_size.x, &window_size.y);
	window_size_atomic().store(window_size);

	glfwSetWindowSizeCallback(gl_state.window, [](GLFWwindow*, int width, int height) {
		if (width == 0 || height == 0) return;
		window_size_atomic().store({ width, height });
	});

	glfwSetWindowPosCallback(gl_state.window, [](GLFWwindow*, int x, int y) {
		window_pos_atomic().store({ x, y });
	});

	static auto get_key_modifiers = [](GLFWwindow* window) {
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

	if (glfwRawMouseMotionSupported()) {
		glfwSetInputMode(gl_state.window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
	}

	glfwSetKeyCallback(gl_state.window, [](GLFWwindow* window, int key, int scancode, int action, int) {
		push_event(rfkt::events::key{ key, scancode, action, get_key_modifiers(window), glfwGetKeyName(key, scancode)});
	});

	glfwSetCursorPosCallback(gl_state.window, [](GLFWwindow* window, double x, double y) {
		push_event(rfkt::events::mouse_move{ x, y });
	});

	glfwSetMouseButtonCallback(gl_state.window, [](GLFWwindow* window, int button, int action, int) {
		push_event(rfkt::events::mouse_button{ button, action, get_key_modifiers(window) });
	});

	glfwSetWindowCloseCallback(gl_state.window, [](GLFWwindow* window) {
		push_event(rfkt::events::window_close{});
	});

	glfwSetCharCallback(gl_state.window, [](GLFWwindow* window, unsigned int codepoint) {
		push_event(rfkt::events::char_input{ codepoint });
	});

	glfwSetScrollCallback(gl_state.window, [](GLFWwindow* window, double x, double y) {
		push_event(rfkt::events::scroll{ x, y });
	});

	glfwSetWindowIconifyCallback(gl_state.window, [](GLFWwindow* window, int iconified) {
		gl_state.minimized = iconified == GLFW_TRUE;
	});

	glfwMakeContextCurrent(gl_state.window);
	glfwSwapInterval(0); // Enable vsync

	if (!gladLoadGL()) {
		SPDLOG_CRITICAL("Could not load OpenGL extensions");
		return false;
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableSetMousePos;
	io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors | ImGuiBackendFlags_HasSetMousePos;
	
	gl_state.cursors[ImGuiMouseCursor_Arrow] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_TextInput] = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_ResizeNS] = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_ResizeEW] = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_Hand] = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_ResizeAll] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
	gl_state.cursors[ImGuiMouseCursor_NotAllowed] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

	io.ConfigWindowsMoveFromTitleBarOnly = true;
	io.Fonts->AddFontFromFileTTF("assets/fonts/Roboto-Medium.ttf", 13);

	ImGui_ImplOpenGL3_Init(glsl_version);

	glfwMakeContextCurrent(nullptr);

	return true;
}

template<typename T>
void event_visitor(T, ImGuiIO&) {
	SPDLOG_INFO("Unhandled event {}", typeid(T).name());
}

ImGuiKey translate_glfw_key(int key, const char* key_name)
{
	if (key >= GLFW_KEY_KP_0 && key <= GLFW_KEY_KP_EQUAL)
		return glfw_to_imgui_key(key);

	if (key_name && key_name[0] != 0 && key_name[1] == 0)
	{
		const char char_names[] = "`-=[]\\,;\'./";
		const int char_keys[] = { GLFW_KEY_GRAVE_ACCENT, GLFW_KEY_MINUS, GLFW_KEY_EQUAL, GLFW_KEY_LEFT_BRACKET, GLFW_KEY_RIGHT_BRACKET, GLFW_KEY_BACKSLASH, GLFW_KEY_COMMA, GLFW_KEY_SEMICOLON, GLFW_KEY_APOSTROPHE, GLFW_KEY_PERIOD, GLFW_KEY_SLASH, 0 };
		IM_ASSERT(IM_ARRAYSIZE(char_names) == IM_ARRAYSIZE(char_keys));
		if (key_name[0] >= '0' && key_name[0] <= '9') { key = GLFW_KEY_0 + (key_name[0] - '0'); }
		else if (key_name[0] >= 'A' && key_name[0] <= 'Z') { key = GLFW_KEY_A + (key_name[0] - 'A'); }
		else if (key_name[0] >= 'a' && key_name[0] <= 'z') { key = GLFW_KEY_A + (key_name[0] - 'a'); }
		else if (const char* p = strchr(char_names, key_name[0])) { key = char_keys[p - char_names]; }
	}

	return glfw_to_imgui_key(key);
}

template<>
void event_visitor<rfkt::events::key>(rfkt::events::key event, ImGuiIO& io) {
	if (event.action != GLFW_PRESS && event.action != GLFW_RELEASE) {
		return;
	}

	io.AddKeyEvent(ImGuiMod_Ctrl, event.mods & GLFW_MOD_CONTROL);
	io.AddKeyEvent(ImGuiMod_Shift, event.mods & GLFW_MOD_SHIFT);
	io.AddKeyEvent(ImGuiMod_Alt, event.mods & GLFW_MOD_ALT);
	io.AddKeyEvent(ImGuiMod_Super, event.mods & GLFW_MOD_SUPER);

	auto keycode = translate_glfw_key(event.key, event.key_name);
	io.AddKeyEvent(keycode, event.action == GLFW_PRESS);
	io.SetKeyEventNativeData(keycode, event.key, event.scancode);
}

template<>
void event_visitor<rfkt::events::mouse_move>(rfkt::events::mouse_move event, ImGuiIO& io) {
	io.AddMousePosEvent((float)event.x, (float)event.y);
}

template<>
void event_visitor<rfkt::events::mouse_button>(rfkt::events::mouse_button event, ImGuiIO& io) {
	io.AddKeyEvent(ImGuiMod_Ctrl, event.mods & GLFW_MOD_CONTROL);
	io.AddKeyEvent(ImGuiMod_Shift, event.mods & GLFW_MOD_SHIFT);
	io.AddKeyEvent(ImGuiMod_Alt, event.mods & GLFW_MOD_ALT);
	io.AddKeyEvent(ImGuiMod_Super, event.mods & GLFW_MOD_SUPER);

	if (event.button >= 0 && event.button < ImGuiMouseButton_COUNT) {
		io.AddMouseButtonEvent(event.button, event.action == GLFW_PRESS);
	}
}

template<>
void event_visitor<rfkt::events::char_input>(rfkt::events::char_input event, ImGuiIO& io) {
	io.AddInputCharacter(event.codepoint);
}

template<>
void event_visitor<rfkt::events::scroll>(rfkt::events::scroll event, ImGuiIO& io) {
	io.AddMouseWheelEvent((float)event.xoffset, (float)event.yoffset);
}

template<>
void event_visitor<rfkt::events::destroy_texture>(rfkt::events::destroy_texture event, ImGuiIO&) {
	CUDA_SAFE_CALL(cuGraphicsUnregisterResource(event.cuda_res));
	glDeleteTextures(1, &event.tex_id);
}

template<>
void event_visitor<rfkt::events::destroy_cuda_map>(rfkt::events::destroy_cuda_map event, ImGuiIO&) {
	CUDA_SAFE_CALL(cuGraphicsUnmapResources(1, &event.cuda_res, 0));
}

void rfkt::gl::begin_frame()
{
	gl_state.frame_start = state::clock_t::now();
	
	if (gl_state.target_fps) {
		auto frame_ns = 1'000'000'000.0/gl_state.target_fps.value();
		gl_state.frame_end = gl_state.frame_start + std::chrono::nanoseconds(static_cast<long long>(frame_ns));
	}

	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	auto& io = ImGui::GetIO();
	auto sz = get_window_size();
	io.DisplaySize = ImVec2(static_cast<float>(sz.x), static_cast<float>(sz.y));

	thread_local decltype(glfwGetTime()) last_time = glfwGetTime();

	auto time = glfwGetTime();
	io.DeltaTime = static_cast<decltype(io.DeltaTime)>(time - last_time);
	last_time = time;

	if (gl_state.cursor_enabled) {
		auto signal = rfkt::signals::set_cursor{};
		signal.cursor = [](ImGuiMouseCursor cursor) -> GLFWcursor* {
			if (cursor == ImGuiMouseCursor_None) return nullptr;
			if (!gl_state.cursors.contains(cursor)) return gl_state.cursors[ImGuiMouseCursor_Arrow];
			return gl_state.cursors[cursor];
		}(ImGui::GetMouseCursor());

		rfkt::push_signal(std::move(signal));
	}
	if (io.WantSetMousePos)
		SPDLOG_INFO("Mouse pos");

	while (true) {
		auto event = poll_event();
		if (!event) break;

		std::visit([](auto&& ev) {
			event_visitor(std::move(ev), ImGui::GetIO());
		}, std::move(event.value()));
	}

	ImGui_ImplOpenGL3_NewFrame();
	ImGui::NewFrame();
}

void preciseSleep(double seconds) {
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

void rfkt::gl::end_frame(bool render)
{
	ImGui::Render();
	if (render && !gl_state.minimized)
	{
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		glfwSwapBuffers(gl_state.window);
	}

	if (gl_state.target_fps.has_value()) {

		auto sleep_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(gl_state.frame_end - std::chrono::steady_clock::now()).count() / 1e9;
		preciseSleep(sleep_seconds);
		//std::this_thread::sleep_until(gl_state.frame_end - std::chrono::milliseconds(1));

		//while (state::clock_t::now() < gl_state.frame_end) { /* busy loop */ }
	}
}

template<typename T>
void signal_visitor(T sig) {
	SPDLOG_INFO("Unhandled signal: {}", typeid(T).name());
}

template<>
void signal_visitor<rfkt::signals::set_cursor>(rfkt::signals::set_cursor sig) {
	auto* window = rfkt::gl::gl_state.window;
	if (sig.cursor == nullptr) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}
	else {
		glfwSetCursor(window, sig.cursor);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

template<>
void signal_visitor<rfkt::signals::set_clipboard>(rfkt::signals::set_clipboard sig) {
	glfwSetClipboardString(rfkt::gl::gl_state.window, sig.contents.c_str());
}

template<>
void signal_visitor<rfkt::signals::get_clipboard>(rfkt::signals::get_clipboard sig) {
	auto* contents = glfwGetClipboardString(rfkt::gl::gl_state.window);
	sig.promise.set_value(contents);
}

template<>
void signal_visitor<rfkt::signals::set_mouse_position>(rfkt::signals::set_mouse_position sig) {
	glfwSetCursorPos(rfkt::gl::gl_state.window, sig.x, sig.y);
}

template<>
void signal_visitor<rfkt::signals::set_cursor_enabled>(rfkt::signals::set_cursor_enabled sig) {
	glfwSetInputMode(rfkt::gl::gl_state.window, GLFW_CURSOR, sig.enabled ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

void rfkt::gl::set_clipboard(std::string contents)
{
	rfkt::push_signal(rfkt::signals::set_clipboard{std::move(contents)});
}

auto rfkt::gl::get_clipboard() -> std::future<std::string> {
	auto sig = rfkt::signals::get_clipboard{};
	sig.promise = std::promise<std::string>{};
	auto future = sig.promise.get_future();
	rfkt::push_signal(std::move(sig));
	return future;
}

void rfkt::gl::set_mouse_position(double x, double y)
{
	rfkt::push_signal(rfkt::signals::set_mouse_position{ x, y});
}

void rfkt::gl::set_cursor_enabled(bool enabled) {
	gl_state.cursor_enabled = enabled;
	rfkt::push_signal(rfkt::signals::set_cursor_enabled{ enabled });
}

bool rfkt::gl::close_requested() {
	return glfwWindowShouldClose(gl_state.window);
}

void rfkt::gl::event_loop(std::stop_token stoke)
{
	while (!stoke.stop_requested()) {
		glfwWaitEvents();

		while (true) {
			auto sig = poll_signal();
			if (!sig) break;

			std::visit([](auto&& s) {
				signal_visitor(std::move(s));
			}, std::move(sig.value()));
		}
	}
}

ImGuiKey glfw_to_imgui_key(int key)
{
	switch (key)
	{
	case GLFW_KEY_TAB: return ImGuiKey_Tab;
	case GLFW_KEY_LEFT: return ImGuiKey_LeftArrow;
	case GLFW_KEY_RIGHT: return ImGuiKey_RightArrow;
	case GLFW_KEY_UP: return ImGuiKey_UpArrow;
	case GLFW_KEY_DOWN: return ImGuiKey_DownArrow;
	case GLFW_KEY_PAGE_UP: return ImGuiKey_PageUp;
	case GLFW_KEY_PAGE_DOWN: return ImGuiKey_PageDown;
	case GLFW_KEY_HOME: return ImGuiKey_Home;
	case GLFW_KEY_END: return ImGuiKey_End;
	case GLFW_KEY_INSERT: return ImGuiKey_Insert;
	case GLFW_KEY_DELETE: return ImGuiKey_Delete;
	case GLFW_KEY_BACKSPACE: return ImGuiKey_Backspace;
	case GLFW_KEY_SPACE: return ImGuiKey_Space;
	case GLFW_KEY_ENTER: return ImGuiKey_Enter;
	case GLFW_KEY_ESCAPE: return ImGuiKey_Escape;
	case GLFW_KEY_APOSTROPHE: return ImGuiKey_Apostrophe;
	case GLFW_KEY_COMMA: return ImGuiKey_Comma;
	case GLFW_KEY_MINUS: return ImGuiKey_Minus;
	case GLFW_KEY_PERIOD: return ImGuiKey_Period;
	case GLFW_KEY_SLASH: return ImGuiKey_Slash;
	case GLFW_KEY_SEMICOLON: return ImGuiKey_Semicolon;
	case GLFW_KEY_EQUAL: return ImGuiKey_Equal;
	case GLFW_KEY_LEFT_BRACKET: return ImGuiKey_LeftBracket;
	case GLFW_KEY_BACKSLASH: return ImGuiKey_Backslash;
	case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
	case GLFW_KEY_GRAVE_ACCENT: return ImGuiKey_GraveAccent;
	case GLFW_KEY_CAPS_LOCK: return ImGuiKey_CapsLock;
	case GLFW_KEY_SCROLL_LOCK: return ImGuiKey_ScrollLock;
	case GLFW_KEY_NUM_LOCK: return ImGuiKey_NumLock;
	case GLFW_KEY_PRINT_SCREEN: return ImGuiKey_PrintScreen;
	case GLFW_KEY_PAUSE: return ImGuiKey_Pause;
	case GLFW_KEY_KP_0: return ImGuiKey_Keypad0;
	case GLFW_KEY_KP_1: return ImGuiKey_Keypad1;
	case GLFW_KEY_KP_2: return ImGuiKey_Keypad2;
	case GLFW_KEY_KP_3: return ImGuiKey_Keypad3;
	case GLFW_KEY_KP_4: return ImGuiKey_Keypad4;
	case GLFW_KEY_KP_5: return ImGuiKey_Keypad5;
	case GLFW_KEY_KP_6: return ImGuiKey_Keypad6;
	case GLFW_KEY_KP_7: return ImGuiKey_Keypad7;
	case GLFW_KEY_KP_8: return ImGuiKey_Keypad8;
	case GLFW_KEY_KP_9: return ImGuiKey_Keypad9;
	case GLFW_KEY_KP_DECIMAL: return ImGuiKey_KeypadDecimal;
	case GLFW_KEY_KP_DIVIDE: return ImGuiKey_KeypadDivide;
	case GLFW_KEY_KP_MULTIPLY: return ImGuiKey_KeypadMultiply;
	case GLFW_KEY_KP_SUBTRACT: return ImGuiKey_KeypadSubtract;
	case GLFW_KEY_KP_ADD: return ImGuiKey_KeypadAdd;
	case GLFW_KEY_KP_ENTER: return ImGuiKey_KeypadEnter;
	case GLFW_KEY_KP_EQUAL: return ImGuiKey_KeypadEqual;
	case GLFW_KEY_LEFT_SHIFT: return ImGuiKey_LeftShift;
	case GLFW_KEY_LEFT_CONTROL: return ImGuiKey_LeftCtrl;
	case GLFW_KEY_LEFT_ALT: return ImGuiKey_LeftAlt;
	case GLFW_KEY_LEFT_SUPER: return ImGuiKey_LeftSuper;
	case GLFW_KEY_RIGHT_SHIFT: return ImGuiKey_RightShift;
	case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
	case GLFW_KEY_RIGHT_ALT: return ImGuiKey_RightAlt;
	case GLFW_KEY_RIGHT_SUPER: return ImGuiKey_RightSuper;
	case GLFW_KEY_MENU: return ImGuiKey_Menu;
	case GLFW_KEY_0: return ImGuiKey_0;
	case GLFW_KEY_1: return ImGuiKey_1;
	case GLFW_KEY_2: return ImGuiKey_2;
	case GLFW_KEY_3: return ImGuiKey_3;
	case GLFW_KEY_4: return ImGuiKey_4;
	case GLFW_KEY_5: return ImGuiKey_5;
	case GLFW_KEY_6: return ImGuiKey_6;
	case GLFW_KEY_7: return ImGuiKey_7;
	case GLFW_KEY_8: return ImGuiKey_8;
	case GLFW_KEY_9: return ImGuiKey_9;
	case GLFW_KEY_A: return ImGuiKey_A;
	case GLFW_KEY_B: return ImGuiKey_B;
	case GLFW_KEY_C: return ImGuiKey_C;
	case GLFW_KEY_D: return ImGuiKey_D;
	case GLFW_KEY_E: return ImGuiKey_E;
	case GLFW_KEY_F: return ImGuiKey_F;
	case GLFW_KEY_G: return ImGuiKey_G;
	case GLFW_KEY_H: return ImGuiKey_H;
	case GLFW_KEY_I: return ImGuiKey_I;
	case GLFW_KEY_J: return ImGuiKey_J;
	case GLFW_KEY_K: return ImGuiKey_K;
	case GLFW_KEY_L: return ImGuiKey_L;
	case GLFW_KEY_M: return ImGuiKey_M;
	case GLFW_KEY_N: return ImGuiKey_N;
	case GLFW_KEY_O: return ImGuiKey_O;
	case GLFW_KEY_P: return ImGuiKey_P;
	case GLFW_KEY_Q: return ImGuiKey_Q;
	case GLFW_KEY_R: return ImGuiKey_R;
	case GLFW_KEY_S: return ImGuiKey_S;
	case GLFW_KEY_T: return ImGuiKey_T;
	case GLFW_KEY_U: return ImGuiKey_U;
	case GLFW_KEY_V: return ImGuiKey_V;
	case GLFW_KEY_W: return ImGuiKey_W;
	case GLFW_KEY_X: return ImGuiKey_X;
	case GLFW_KEY_Y: return ImGuiKey_Y;
	case GLFW_KEY_Z: return ImGuiKey_Z;
	case GLFW_KEY_F1: return ImGuiKey_F1;
	case GLFW_KEY_F2: return ImGuiKey_F2;
	case GLFW_KEY_F3: return ImGuiKey_F3;
	case GLFW_KEY_F4: return ImGuiKey_F4;
	case GLFW_KEY_F5: return ImGuiKey_F5;
	case GLFW_KEY_F6: return ImGuiKey_F6;
	case GLFW_KEY_F7: return ImGuiKey_F7;
	case GLFW_KEY_F8: return ImGuiKey_F8;
	case GLFW_KEY_F9: return ImGuiKey_F9;
	case GLFW_KEY_F10: return ImGuiKey_F10;
	case GLFW_KEY_F11: return ImGuiKey_F11;
	case GLFW_KEY_F12: return ImGuiKey_F12;
	default: return ImGuiKey_None;
	}
}