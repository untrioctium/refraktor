#include <string>
#include <future>
#include <variant>
#include <optional>

#include "imftw_internal.h"
#include <imftw/signals.h>


namespace imftw::sigs::glfw {

    struct set_cursor {
        GLFWcursor* cursor;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_cursor_position {
        double x;
        double y;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_cursor_enabled {
        bool enabled;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_window_title {
        std::string title;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_window_position {
        int x;
        int y;

        mutable std::promise<void> done;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_window_size {
        int width;
        int height;

        mutable std::promise<void> done;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_window_visible {
        bool visible;

        mutable std::promise<void> done;

		void handle(imftw::context_t& ctx) const;
	};

    struct set_window_decorated {
        bool decorated;

        mutable std::promise<void> done;

		void handle(imftw::context_t& ctx) const;
	};

    struct set_window_maximized {
        bool maximized;

		mutable std::promise<void> done;

        void handle(imftw::context_t& ctx) const;
    };

    struct get_monitor_size {
        mutable std::promise<std::pair<int, int>> promise;

		void handle(imftw::context_t& ctx) const;
	};

    struct get_monitor_position {
		mutable std::promise<std::pair<int, int>> promise;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_window_progress_mode {
        imftw::sig::progress_mode mode;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_window_progress_value {
        unsigned long long completed;
        unsigned long long total;

        void handle(imftw::context_t& ctx) const;
    };

    struct set_clipboard {
        std::string contents;

        void handle(imftw::context_t& ctx) const;
    };

    struct get_clipboard {
       mutable std::promise<std::string> promise;

       void handle(imftw::context_t& ctx) const;
    };
}

namespace imftw::sigs::opengl {
    struct set_vsync_enabled {
		bool enabled;

		void handle(imftw::context_t& ctx) const;
	};

    struct set_low_power_mode {
		bool enabled;

		void handle(imftw::context_t& ctx) const;
	};

    struct set_target_framerate {
		unsigned int fps;

		void handle(imftw::context_t& ctx) const;
	};
}

namespace imftw::sig {

    using glfw_signal = std::variant<
        imftw::sigs::glfw::set_cursor,
        imftw::sigs::glfw::set_cursor_position,
        imftw::sigs::glfw::set_cursor_enabled,
        imftw::sigs::glfw::set_window_title,
        imftw::sigs::glfw::set_window_position,
        imftw::sigs::glfw::set_window_size,
        imftw::sigs::glfw::set_window_visible,
        imftw::sigs::glfw::set_window_decorated,
        imftw::sigs::glfw::set_window_maximized,
        imftw::sigs::glfw::get_monitor_size,
        imftw::sigs::glfw::get_monitor_position,
        imftw::sigs::glfw::set_window_progress_mode,
        imftw::sigs::glfw::set_window_progress_value,
        imftw::sigs::glfw::set_clipboard,
        imftw::sigs::glfw::get_clipboard>;

    void push_glfw_signal(glfw_signal&& signal);
    std::optional<glfw_signal> poll_glfw_signal();

    using opengl_signal = std::variant<
        imftw::sigs::opengl::set_vsync_enabled,
		imftw::sigs::opengl::set_low_power_mode,
		imftw::sigs::opengl::set_target_framerate>;

    void push_opengl_signal(opengl_signal&& signal);
	std::optional<opengl_signal> poll_opengl_signal();
}