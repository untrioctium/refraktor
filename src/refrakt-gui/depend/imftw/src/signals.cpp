#include <concurrentqueue.h>
#include <blockingconcurrentqueue.h>

#include "signals_internal.h"

// glfw signal implementations
namespace imftw::sig::glfw {
	using signal_queue_t = moodycamel::ConcurrentQueue<imftw::sig::glfw_signal, moodycamel::ConcurrentQueueDefaultTraits>;

	signal_queue_t& signal_queue() {
		static signal_queue_t queue;
		return queue;
	}
}

void imftw::sig::push_glfw_signal(imftw::sig::glfw_signal&& signal) {
	 glfw::signal_queue().enqueue(std::move(signal));
	 glfwPostEmptyEvent(); // make sure the main thread wakes and processes this signal
}

std::optional<imftw::sig::glfw_signal> imftw::sig::poll_glfw_signal() {
	imftw::sig::glfw_signal signal;
	if (glfw::signal_queue().try_dequeue(signal)) {
		return signal;
	}
	return std::nullopt;
}

// opengl signal implementations
namespace imftw::sig::opengl {
	using signal_queue_t = moodycamel::ConcurrentQueue<imftw::sig::opengl_signal, moodycamel::ConcurrentQueueDefaultTraits>;

	signal_queue_t& signal_queue() {
		static signal_queue_t queue;
		return queue;
	}
}

void imftw::sig::push_opengl_signal(imftw::sig::opengl_signal&& signal) {
	 opengl::signal_queue().enqueue(std::move(signal));
}

std::optional<imftw::sig::opengl_signal> imftw::sig::poll_opengl_signal() {
	imftw::sig::opengl_signal signal;
	if (opengl::signal_queue().try_dequeue(signal)) {
		return signal;
	}
	return std::nullopt;
}

// set_cursor
void imftw::sigs::glfw::set_cursor::handle(imftw::context_t& ctx) const {
	if (cursor == nullptr) {
		glfwSetInputMode(ctx.window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}
	else {
		glfwSetCursor(ctx.window, cursor);
		glfwSetInputMode(ctx.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void imftw::sig::set_cursor(ImGuiMouseCursor cursor) {
	auto& ctx = context();

	auto glfw_cursor_iter = ctx.cursors.find(cursor);
	if (glfw_cursor_iter == ctx.cursors.end()) return;

	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_cursor{ glfw_cursor_iter->second });
}

// set_cursor_position
void imftw::sigs::glfw::set_cursor_position::handle(imftw::context_t& ctx) const {
	glfwSetCursorPos(ctx.window, x, y);
}

void imftw::sig::set_cursor_position(double x, double y) {
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_cursor_position{ x, y });
}

// set_cursor_enabled
void imftw::sigs::glfw::set_cursor_enabled::handle(imftw::context_t& ctx) const {
	glfwSetInputMode(ctx.window, GLFW_CURSOR, enabled ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

void imftw::sig::set_cursor_enabled(bool enabled) {
	context().cursor_enabled = enabled;
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_cursor_enabled{ enabled });
}

// set_window_title
void imftw::sigs::glfw::set_window_title::handle(imftw::context_t& ctx) const {
	glfwSetWindowTitle(ctx.window, title.c_str());
}

void imftw::sig::set_window_title(std::string title) {
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_title{ std::move(title) });
}

// set_window_position
void imftw::sigs::glfw::set_window_position::handle(imftw::context_t& ctx) const {
	glfwSetWindowPos(ctx.window, x, y);
	ctx.window_position = { x, y };
	done.set_value();
}

std::future<void> imftw::sig::set_window_position(int x, int y) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_position{ x, y, std::move(promise)});
	return future;
}

// set_window_size
void imftw::sigs::glfw::set_window_size::handle(imftw::context_t& ctx) const {
	glfwSetWindowSize(ctx.window, width, height);
	ctx.window_size = { width, height };
	done.set_value();
}

std::future<void> imftw::sig::set_window_size(int width, int height) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_size{ width, height, std::move(promise)});
	return future;
}

// set_window_visible
void imftw::sigs::glfw::set_window_visible::handle(imftw::context_t& ctx) const {
	if (visible) {
		glfwShowWindow(ctx.window);
	}
	else {
		glfwHideWindow(ctx.window);
	}
	done.set_value();
}

std::future<void> imftw::sig::set_window_visible(bool visible) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_visible{ visible, std::move(promise)});
	return future;
}

// set_window_decorated
void imftw::sigs::glfw::set_window_decorated::handle(imftw::context_t& ctx) const {
	glfwSetWindowAttrib(ctx.window, GLFW_DECORATED, decorated);
	done.set_value();
}

std::future<void> imftw::sig::set_window_decorated(bool decorated) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_decorated{ decorated, std::move(promise)});
	return future;
}

// set_window_maximized
void imftw::sigs::glfw::set_window_maximized::handle(imftw::context_t& ctx) const {
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

std::future<void> imftw::sig::set_window_maximized(bool maximized) {
	auto promise = std::promise<void>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_maximized{ maximized, std::move(promise)});
	return future;
}

// get_monitor_size
void imftw::sigs::glfw::get_monitor_size::handle(imftw::context_t& ctx) const {
	auto mode = glfwGetVideoMode(ctx.monitor);
	promise.set_value({mode->width, mode->height});
}

std::future<std::pair<int, int>> imftw::sig::get_monitor_size() {
	auto promise = std::promise<std::pair<int, int>>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::get_monitor_size{ std::move(promise)});
	return future;
}

// get_monitor_position
void imftw::sigs::glfw::get_monitor_position::handle(imftw::context_t& ctx) const {
	int xpos, ypos;
	glfwGetMonitorPos(ctx.monitor, &xpos, &ypos);
	promise.set_value({xpos, ypos});
}

std::future<std::pair<int, int>> imftw::sig::get_monitor_position() {
	auto promise = std::promise<std::pair<int, int>>{};
	auto future = promise.get_future();
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::get_monitor_position{ std::move(promise)});
	return future;
}

// set_window_progress_mode
void imftw::sigs::glfw::set_window_progress_mode::handle(imftw::context_t& ctx) const {
#ifdef _WIN32
	if (!ctx.taskbar) return;

	auto windows_mode = [](imftw::sig::progress_mode mode) {
		switch (mode) {
		case imftw::sig::progress_mode::indeterminate:
			return TBPF_INDETERMINATE;
		case imftw::sig::progress_mode::determinate:
			return TBPF_NORMAL;
		case imftw::sig::progress_mode::error:
			return TBPF_ERROR;
		case imftw::sig::progress_mode::paused:
			return TBPF_PAUSED;
		case imftw::sig::progress_mode::disabled:
			return TBPF_NOPROGRESS;
		default:
			return TBPF_NOPROGRESS;
		}
	}(mode);

	ctx.taskbar->SetProgressState(ctx.hwnd, windows_mode);
#endif
}

void imftw::sig::set_window_progress_mode(imftw::sig::progress_mode mode) {
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_progress_mode{ mode });
}

// set_window_progress_value
void imftw::sigs::glfw::set_window_progress_value::handle(imftw::context_t& ctx) const {
#ifdef _WIN32
	if (!ctx.taskbar) return;

	ctx.taskbar->SetProgressValue(ctx.hwnd, completed, total);
#endif
}

void imftw::sig::set_window_progress_value(unsigned long long completed, unsigned long long total) {
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_window_progress_value{ completed, total });
}

// set_clipboard
void imftw::sigs::glfw::set_clipboard::handle(imftw::context_t& ctx) const {
	glfwSetClipboardString(ctx.window, contents.c_str());
}

void imftw::sig::set_clipboard(std::string contents) {
	imftw::sig::push_glfw_signal(imftw::sigs::glfw::set_clipboard{ std::move(contents) });
}

// get_clipboard
void imftw::sigs::glfw::get_clipboard::handle(imftw::context_t& ctx) const {
	promise.set_value(glfwGetClipboardString(ctx.window));
}

std::future<std::string> imftw::sig::get_clipboard() {
	std::promise<std::string> promise;
	auto future = promise.get_future();

	imftw::sig::push_glfw_signal(imftw::sigs::glfw::get_clipboard{ std::move(promise) });

	return future;
}

// set_vsync_enabled
void imftw::sigs::opengl::set_vsync_enabled::handle(imftw::context_t& ctx) const {
	glfwSwapInterval(enabled ? 1 : 0);
}

void imftw::sig::set_vsync_enabled(bool enabled) {
	imftw::sig::push_opengl_signal(imftw::sigs::opengl::set_vsync_enabled{ enabled });
}

// set_low_power_mode
void imftw::sigs::opengl::set_low_power_mode::handle(imftw::context_t& ctx) const {
	ctx.low_power_mode = enabled;
}

void imftw::sig::set_low_power_mode(bool enabled) {
	imftw::sig::push_opengl_signal(imftw::sigs::opengl::set_low_power_mode{ enabled });
}

// set_target_framerate
void imftw::sigs::opengl::set_target_framerate::handle(imftw::context_t& ctx) const {
	if(fps > 0)
		ctx.target_framerate = fps;
	else
		ctx.target_framerate = std::nullopt;
}

void imftw::sig::set_target_framerate(unsigned int fps) {
	imftw::sig::push_opengl_signal(imftw::sigs::opengl::set_target_framerate{ fps });
}
