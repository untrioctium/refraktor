#pragma once

#include <functional>
#include <imgui.h>
#include <string>

#include <filesystem>

namespace imftw {

    void run(std::string_view window_title, std::move_only_function<void()>&& main_function);

    void begin_frame(ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
    void end_frame(bool render = true);

    std::string show_open_dialog(const std::filesystem::path& path, std::string_view filter);
    std::string show_save_dialog(const std::filesystem::path& path, std::string_view filter);

    void request_frame();
}