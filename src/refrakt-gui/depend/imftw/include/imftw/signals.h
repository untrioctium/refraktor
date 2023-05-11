#pragma once

#include <string>
#include <future>

#include <imgui.h>

namespace imftw::sig {

    void set_clipboard(std::string contents);
    inline void set_clipboard(std::string_view contents) {
        set_clipboard(std::string(contents));
    }

    auto get_clipboard() -> std::future<std::string>;

    void set_cursor_position(double x, double y);
    void set_cursor_enabled(bool);
    void set_cursor(ImGuiMouseCursor cursor);

    void set_window_title(std::string title);
    inline void set_window_title(std::string_view title) {
        set_window_title(std::string(title));
    }

    auto set_window_size(int width, int height) -> std::future<void>;
    auto set_window_position(int x, int y) -> std::future<void>;
    auto set_window_visible(bool) -> std::future<void>;
    auto set_window_decorated(bool) -> std::future<void>;
    auto set_window_maximized(bool) -> std::future<void>;

    auto get_monitor_size() -> std::future<std::pair<int, int>>;
    auto get_monitor_position() -> std::future<std::pair<int, int>>;

    enum class progress_mode {
        indeterminate,
        determinate,
        error,
        paused,
        disabled
    };
    
    void set_window_progress_mode(progress_mode mode);
    void set_window_progress_value(unsigned long long completed, unsigned long long total);

    void set_vsync_enabled(bool);
    void set_low_power_mode(bool);
    void set_target_framerate(unsigned int fps);


}