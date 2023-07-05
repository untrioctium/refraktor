#pragma once

#include <string>
#include <future>

#include <imgui.h>

namespace ImFtw::Sig {

    void SetClipboard(std::string contents);
    inline void SetClipboard(std::string_view contents) {
        SetClipboard(std::string(contents));
    }

    auto GetClipboard() -> std::future<std::string>;

    void SetCursorPosition(double x, double y);
    void SetCursorEnabled(bool);
    void SetCursor(ImGuiMouseCursor cursor);

    void SetWindowTitle(std::string title);
    inline void SetWindowTitle(std::string_view title) {
        SetWindowTitle(std::string(title));
    }

    auto SetWindowSize(int width, int height) -> std::future<void>;
    auto SetWindowPosition(int x, int y) -> std::future<void>;
    auto SetWindowVisible(bool) -> std::future<void>;
    auto SetWindowDecorated(bool) -> std::future<void>;
    auto SetWindowMaximized(bool) -> std::future<void>;

    auto GetMonitorSize() -> std::future<std::pair<int, int>>;
    auto GetMonitorPosition() -> std::future<std::pair<int, int>>;

    enum class ProgressMode {
        Indeterminate,
        Determinate, 
        Error,
        Paused,
        Disabled
    };
    
    void SetWindowProgressMode(ProgressMode mode);
    void SetWindowProgressValue(unsigned long long completed, unsigned long long total);

    void SetVSyncEnabled(bool);
    void SetLowPowerMode(bool);
    void SetTargetFramerate(unsigned int fps);


}