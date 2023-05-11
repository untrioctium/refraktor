#include <variant>
#include <optional>

#include <readerwriterqueue.h>

struct ImGuiIO;

namespace imftw::events {
    struct mouse_move {
        double x;
        double y;

        void handle(ImGuiIO& io) const;
    };

    struct mouse_button {
        int button;
        int action;
        int mods;

        void handle(ImGuiIO& io) const;
    };
    
    struct mouse_scroll {
        double x;
        double y;

        void handle(ImGuiIO& io) const;
    };

    struct key {
        int key;
        int scancode;
        int action;
        int mods;
        const char* key_name;

        void handle(ImGuiIO& io) const;
    };

    struct char_input {
        unsigned int codepoint;

        void handle(ImGuiIO& io) const;
    };

    struct wakeup {
        void handle(ImGuiIO&) const { /* no-op; only waking the render thread if asleep */ }
    };
}

namespace imftw {
    using event = std::variant<
        events::mouse_move,
        events::mouse_button,
        events::mouse_scroll,
        events::key,
        events::char_input,
        events::wakeup
    >;

    void push_event(event&& e);
    std::optional<event> poll_event();
}