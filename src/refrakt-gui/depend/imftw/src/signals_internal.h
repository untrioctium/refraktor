#include <string>
#include <future>
#include <variant>
#include <optional>

#include "imftw_internal.h"
#include <imftw/signals.h>

namespace ImFtw::sigs::glfw2 {

    template<typename T>
    struct signal {
        using result_t = decltype(T::handle(std::declval<ImFtw::context_t&>()));

        std::promise<result_t> done;


    };

}

namespace ImFtw::sigs::glfw {

    struct set_cursor {
        GLFWcursor* cursor;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_cursor_position {
        double x;
        double y;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_cursor_enabled {
        bool enabled;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_window_title {
        std::string title;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_window_position {
        int x;
        int y;

        mutable std::promise<void> done;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_window_size {
        int width;
        int height;

        mutable std::promise<void> done;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_window_visible {
        bool visible;

        mutable std::promise<void> done;

		void handle(ImFtw::context_t& ctx) const;
	};

    struct set_window_decorated {
        bool decorated;

        mutable std::promise<void> done;

		void handle(ImFtw::context_t& ctx) const;
	};

    struct set_window_maximized {
        bool maximized;

		mutable std::promise<void> done;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct get_monitor_size {
        mutable std::promise<std::pair<int, int>> promise;

		void handle(ImFtw::context_t& ctx) const;
	};

    struct get_monitor_position {
		mutable std::promise<std::pair<int, int>> promise;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_window_progress_mode {
        ImFtw::Sig::ProgressMode mode;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_window_progress_value {
        unsigned long long completed;
        unsigned long long total;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct set_clipboard {
        std::string contents;

        void handle(ImFtw::context_t& ctx) const;
    };

    struct get_clipboard {
       mutable std::promise<std::string> promise;

       void handle(ImFtw::context_t& ctx) const;
    };
}

namespace ImFtw::sigs::opengl {
    struct set_vsync_enabled {
		bool enabled;

		void handle(ImFtw::context_t& ctx) const;
	};

    struct set_low_power_mode {
		bool enabled;

		void handle(ImFtw::context_t& ctx) const;
	};

    struct set_target_framerate {
		unsigned int fps;

		void handle(ImFtw::context_t& ctx) const;
	};

    struct set_imgui_ini_path {
        std::string path;

        void handle(ImFtw::context_t& ctx) const;
    };
}

namespace ImFtw::Sig {

    using glfw_signal = std::variant<
        ImFtw::sigs::glfw::set_cursor,
        ImFtw::sigs::glfw::set_cursor_position,
        ImFtw::sigs::glfw::set_cursor_enabled,
        ImFtw::sigs::glfw::set_window_title,
        ImFtw::sigs::glfw::set_window_position,
        ImFtw::sigs::glfw::set_window_size,
        ImFtw::sigs::glfw::set_window_visible,
        ImFtw::sigs::glfw::set_window_decorated,
        ImFtw::sigs::glfw::set_window_maximized,
        ImFtw::sigs::glfw::get_monitor_size,
        ImFtw::sigs::glfw::get_monitor_position,
        ImFtw::sigs::glfw::set_window_progress_mode,
        ImFtw::sigs::glfw::set_window_progress_value,
        ImFtw::sigs::glfw::set_clipboard,
        ImFtw::sigs::glfw::get_clipboard>;

    void push_glfw_signal(glfw_signal&& signal);
    std::optional<glfw_signal> poll_glfw_signal();

    using opengl_signal = std::variant<
        ImFtw::sigs::opengl::set_vsync_enabled,
		ImFtw::sigs::opengl::set_low_power_mode,
		ImFtw::sigs::opengl::set_target_framerate,
        ImFtw::sigs::opengl::set_imgui_ini_path>;

    void push_opengl_signal(opengl_signal&& signal);
	std::optional<opengl_signal> poll_opengl_signal();
}