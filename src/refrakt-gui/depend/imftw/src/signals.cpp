#include <concurrentqueue.h>
#include <blockingconcurrentqueue.h>

#include "signals_internal.h"

// glfw signal implementations
namespace ImFtw::Sig::glfw {
	using signal_queue_t = moodycamel::ConcurrentQueue<ImFtw::Sig::glfw_signal, moodycamel::ConcurrentQueueDefaultTraits>;

	signal_queue_t& signal_queue() {
		static signal_queue_t queue;
		return queue;
	}
}

void ImFtw::Sig::push_glfw_signal(ImFtw::Sig::glfw_signal&& signal) {
	 glfw::signal_queue().enqueue(std::move(signal));
	 glfwPostEmptyEvent(); // make sure the main thread wakes and processes this signal
}

std::optional<ImFtw::Sig::glfw_signal> ImFtw::Sig::poll_glfw_signal() {
	ImFtw::Sig::glfw_signal signal;
	if (glfw::signal_queue().try_dequeue(signal)) {
		return signal;
	}
	return std::nullopt;
}

// opengl signal implementations
namespace ImFtw::Sig::opengl {
	using signal_queue_t = moodycamel::ConcurrentQueue<ImFtw::Sig::opengl_signal, moodycamel::ConcurrentQueueDefaultTraits>;

	signal_queue_t& signal_queue() {
		static signal_queue_t queue;
		return queue;
	}
}

void ImFtw::Sig::push_opengl_signal(ImFtw::Sig::opengl_signal&& signal) {
	 opengl::signal_queue().enqueue(std::move(signal));
}

std::optional<ImFtw::Sig::opengl_signal> ImFtw::Sig::poll_opengl_signal() {
	ImFtw::Sig::opengl_signal signal;
	if (opengl::signal_queue().try_dequeue(signal)) {
		return signal;
	}
	return std::nullopt;
}

// set_cursor
void ImFtw::sigs::glfw::set_cursor::handle(ImFtw::context_t& ctx) const {
	if (cursor == nullptr) {
		glfwSetInputMode(ctx.window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}
	else {
		glfwSetCursor(ctx.window, cursor);
		glfwSetInputMode(ctx.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void ImFtw::Sig::SetCursor(ImGuiMouseCursor cursor) {
	auto& ctx = context();

	auto glfw_cursor_iter = ctx.cursors.find(cursor);
	if (glfw_cursor_iter == ctx.cursors.end()) return;

	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_cursor{ glfw_cursor_iter->second });
}

// set_cursor_position
void ImFtw::sigs::glfw::set_cursor_position::handle(ImFtw::context_t& ctx) const {
	glfwSetCursorPos(ctx.window, x, y);
}

void ImFtw::Sig::SetCursorPosition(double x, double y) {
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_cursor_position{ x, y });
}

// set_cursor_enabled
void ImFtw::sigs::glfw::set_cursor_enabled::handle(ImFtw::context_t& ctx) const {
	glfwSetInputMode(ctx.window, GLFW_CURSOR, enabled ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

void ImFtw::Sig::SetCursorEnabled(bool enabled) {
	context().cursor_enabled = enabled;
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_cursor_enabled{ enabled });
}

// set_window_title
void ImFtw::sigs::glfw::set_window_title::handle(ImFtw::context_t& ctx) const {
	glfwSetWindowTitle(ctx.window, title.c_str());
}

void ImFtw::Sig::SetWindowTitle(std::string title) {
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_title{ std::move(title) });
}

// set_window_position
void ImFtw::sigs::glfw::set_window_position::handle(ImFtw::context_t& ctx) const {
	glfwSetWindowPos(ctx.window, x, y);
	ctx.window_position = { x, y };
	done.set_value();
}

std::future<void> ImFtw::Sig::SetWindowPosition(int x, int y) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_position{ x, y, std::move(promise)});
	return future;
}

// set_window_size
void ImFtw::sigs::glfw::set_window_size::handle(ImFtw::context_t& ctx) const {
	glfwSetWindowSize(ctx.window, width, height);
	ctx.window_size = { width, height };
	done.set_value();
}

std::future<void> ImFtw::Sig::SetWindowSize(int width, int height) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_size{ width, height, std::move(promise)});
	return future;
}

// set_window_visible
void ImFtw::sigs::glfw::set_window_visible::handle(ImFtw::context_t& ctx) const {
	if (visible) {
		glfwShowWindow(ctx.window);
	}
	else {
		glfwHideWindow(ctx.window);
	}
	done.set_value();
}

std::future<void> ImFtw::Sig::SetWindowVisible(bool visible) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_visible{ visible, std::move(promise)});
	return future;
}

// set_window_decorated
void ImFtw::sigs::glfw::set_window_decorated::handle(ImFtw::context_t& ctx) const {
	glfwSetWindowAttrib(ctx.window, GLFW_DECORATED, decorated);
	done.set_value();
}

std::future<void> ImFtw::Sig::SetWindowDecorated(bool decorated) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_decorated{ decorated, std::move(promise)});
	return future;
}

// set_window_maximized
void ImFtw::sigs::glfw::set_window_maximized::handle(ImFtw::context_t& ctx) const {
	if (maximized) {
		glfwMaximizeWindow(ctx.window);
	}
	else {
		glfwRestoreWindow(ctx.window);
	}
	int width, height;
	glfwGetWindowSize(ctx.window, &width, &height);
	ctx.window_size = { width, height };
	done.set_value();
}

std::future<void> ImFtw::Sig::SetWindowMaximized(bool maximized) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_maximized{ maximized, std::move(promise)});
	return future;
}

// get_monitor_size
void ImFtw::sigs::glfw::get_monitor_size::handle(ImFtw::context_t& ctx) const {
	auto mode = glfwGetVideoMode(ctx.monitor);
	promise.set_value({mode->width, mode->height});
}

std::future<std::pair<int, int>> ImFtw::Sig::GetMonitorSize() {
	auto promise = std::promise<std::pair<int, int>>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::get_monitor_size{ std::move(promise)});
	return future;
}

// get_monitor_position
void ImFtw::sigs::glfw::get_monitor_position::handle(ImFtw::context_t& ctx) const {
	int xpos, ypos;
	glfwGetMonitorPos(ctx.monitor, &xpos, &ypos);
	promise.set_value({xpos, ypos});
}

std::future<std::pair<int, int>> ImFtw::Sig::GetMonitorPosition() {
	auto promise = std::promise<std::pair<int, int>>{};
	auto future = promise.get_future();
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::get_monitor_position{ std::move(promise)});
	return future;
}

// set_window_progress_mode
void ImFtw::sigs::glfw::set_window_progress_mode::handle(ImFtw::context_t& ctx) const {
#ifdef _WIN32
	if (!ctx.taskbar) return;

	auto windows_mode = [](ImFtw::Sig::ProgressMode mode) {
		switch (mode) {
		case ImFtw::Sig::ProgressMode::Indeterminate:
			return TBPF_INDETERMINATE;
		case ImFtw::Sig::ProgressMode::Determinate:
			return TBPF_NORMAL;
		case ImFtw::Sig::ProgressMode::Error:
			return TBPF_ERROR;
		case ImFtw::Sig::ProgressMode::Paused:
			return TBPF_PAUSED;
		case ImFtw::Sig::ProgressMode::Disabled:
			return TBPF_NOPROGRESS;
		default:
			return TBPF_NOPROGRESS;
		}
	}(mode);

	ctx.taskbar->SetProgressState(ctx.hwnd, windows_mode);
#endif
}

void ImFtw::Sig::SetWindowProgressMode(ImFtw::Sig::ProgressMode mode) {
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_progress_mode{ mode });
}

// set_window_progress_value
void ImFtw::sigs::glfw::set_window_progress_value::handle(ImFtw::context_t& ctx) const {
#ifdef _WIN32
	if (!ctx.taskbar) return;

	ctx.taskbar->SetProgressValue(ctx.hwnd, completed, total);
#endif
}

void ImFtw::Sig::SetWindowProgressValue(unsigned long long completed, unsigned long long total) {
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_window_progress_value{ completed, total });
}

// set_clipboard
void ImFtw::sigs::glfw::set_clipboard::handle(ImFtw::context_t& ctx) const {
	glfwSetClipboardString(ctx.window, contents.c_str());
}

void ImFtw::Sig::SetClipboard(std::string contents) {
	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::set_clipboard{ std::move(contents) });
}

// get_clipboard
void ImFtw::sigs::glfw::get_clipboard::handle(ImFtw::context_t& ctx) const {
	promise.set_value(glfwGetClipboardString(ctx.window));
}

std::future<std::string> ImFtw::Sig::GetClipboard() {
	std::promise<std::string> promise;
	auto future = promise.get_future();

	ImFtw::Sig::push_glfw_signal(ImFtw::sigs::glfw::get_clipboard{ std::move(promise) });

	return future;
}

// set_vsync_enabled
void ImFtw::sigs::opengl::set_vsync_enabled::handle(ImFtw::context_t& ctx) const {
	glfwSwapInterval(enabled ? 1 : 0);
}

void ImFtw::Sig::SetVSyncEnabled(bool enabled) {
	ImFtw::Sig::push_opengl_signal(ImFtw::sigs::opengl::set_vsync_enabled{ enabled });
}

// set_low_power_mode
void ImFtw::sigs::opengl::set_low_power_mode::handle(ImFtw::context_t& ctx) const {
	ctx.low_power_mode = enabled;
}

void ImFtw::Sig::SetLowPowerMode(bool enabled) {
	ImFtw::Sig::push_opengl_signal(ImFtw::sigs::opengl::set_low_power_mode{ enabled });
}

// set_target_framerate
void ImFtw::sigs::opengl::set_target_framerate::handle(ImFtw::context_t& ctx) const {
	if(fps > 0)
		ctx.target_framerate = fps;
	else
		ctx.target_framerate = std::nullopt;
}

void ImFtw::sigs::opengl::set_imgui_ini_path::handle(ImFtw::context_t& ctx) const {
	ctx.imgui_ini_path = path;
	ImGui::GetIO().IniFilename = ctx.imgui_ini_path.c_str();
}

void ImFtw::Sig::SetTargetFramerate(unsigned int fps) {
	ImFtw::Sig::push_opengl_signal(ImFtw::sigs::opengl::set_target_framerate{ fps });
}
