#include <chrono>
#include <optional>
#include <unordered_map>
#include <atomic>

#include <imgui.h>
#include <imgui_internal.h>
#include <concurrentqueue.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <ShObjIdl_core.h>
#include <dwmapi.h>
#endif

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace ImFtw {

    struct context_t {

        context_t() = default;

        context_t(const context_t&) = delete;
        context_t& operator=(const context_t&) = delete;

        context_t(context_t&&) = delete;
        context_t& operator=(context_t&&) = delete;

        using clock_t = std::chrono::steady_clock;
        using timepoint_t = clock_t::time_point;

        GLFWwindow* window;
        GLFWmonitor* monitor;

        std::optional<unsigned int> target_framerate;
        bool low_power_mode = false;
        bool vsync_enabled = false;

        timepoint_t frame_start;
        timepoint_t frame_end_target;

        ImVec2 framebuffer_size;

        std::unordered_map<ImGuiMouseCursor, GLFWcursor*> cursors;
        std::atomic_bool cursor_enabled = true;

        std::atomic_bool maximized;
        std::atomic_bool iconified;

        std::atomic<std::pair<int, int>> window_size;
        std::atomic<std::pair<int, int>> window_position;

        decltype(glfwGetTime()) last_time;

        std::thread::id opengl_thread_id;

        moodycamel::ConcurrentQueue<std::move_only_function<void()>> deferred_functions;

        std::string imgui_ini_path;

#ifdef _WIN32
        HWND hwnd;
        ITaskbarList4* taskbar;
#endif

    };

    context_t& context();

}