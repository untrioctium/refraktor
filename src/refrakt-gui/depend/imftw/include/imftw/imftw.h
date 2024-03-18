#pragma once

#include <functional>
#include <imgui.h>
#include <string>

#include <filesystem>

#include <span>

namespace ImFtw {

    int Run(std::string_view window_title, std::string_view ini_path, std::move_only_function<int()>&& main_function);

    void BeginFrame(ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
    void EndFrame(bool render = true);

    void DeferNextFrame(std::move_only_function<void()>&&);

    template<typename T>
    auto MakeDeferer(T&& func) {
        return [f = std::forward<T>(func)]() mutable {
			DeferNextFrame(std::move(f));
		};
	}

    bool OnRenderingThread();

    std::string ShowOpenDialog(const std::filesystem::path& path, std::string_view filter);
    std::string ShowSaveDialog(const std::filesystem::path& path, std::string_view filter);

    void OpenBrowser(std::string_view url);

    double Time();

    void RequestFrame();
}