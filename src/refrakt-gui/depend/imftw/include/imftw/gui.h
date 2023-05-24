#pragma once

#include <imgui.h>
#include <imgui_internal.h>

#include <imftw/signals.h>

namespace imftw {

    namespace detail {
        template<typename T>
        ImGuiDataType_ get_data_type_of() {
            if constexpr (std::same_as<T, std::int8_t>) {
                return ImGuiDataType_S8;
            }
            else if constexpr (std::same_as<T, std::uint8_t>) {
                return ImGuiDataType_U8;
            }
            else if constexpr (std::same_as<T, std::int16_t>) {
                return ImGuiDataType_S16;
            }
            else if constexpr (std::same_as<T, std::uint16_t>) {
                return ImGuiDataType_U16;
            }
            else if constexpr (std::same_as<T, std::int32_t>) {
                return ImGuiDataType_S32;
            }
            else if constexpr (std::same_as<T, std::uint32_t>) {
                return ImGuiDataType_U32;
            }
            else if constexpr (std::same_as<T, std::int64_t>) {
                return ImGuiDataType_S64;
            }
            else if constexpr (std::same_as<T, std::uint64_t>) {
                return ImGuiDataType_U64;
            }
            else if constexpr (std::same_as<T, float>) {
                return ImGuiDataType_Float;
            }
            else if constexpr (std::same_as<T, double>) {
                return ImGuiDataType_Double;
            }
            else {
                static_assert("Unsupported type for imftw::detail::get_data_type_of()");
            }
        }


    }

    inline void tooltip(std::string_view text, bool show_icon = true) {
        if (show_icon) ImGui::Text("\xee\xa2\x87");
        if (ImGui::IsItemHovered())
        {
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(text.data());
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
    }

    inline void text_centered(std::string_view text) {
        auto region = ImGui::GetContentRegionAvail();
        auto size = ImGui::CalcTextSize(text.data());
        ImGui::SetCursorPosX(region.x / 2 - size.x / 2);
        ImGui::TextUnformatted(text.data());
    }

    template<typename T>
    concept draggable_type = 
        std::same_as<T, std::int8_t> 
        || std::same_as<T, std::uint8_t> 
        || std::same_as<T, std::int16_t> 
        || std::same_as<T, std::uint16_t> 
        || std::same_as<T, std::int32_t>
        || std::same_as<T, std::uint32_t>
        || std::same_as<T, std::int64_t>
        || std::same_as<T, std::uint64_t>
        || std::same_as<T, float>
        || std::same_as<T, double>;

    template<draggable_type ValueType>
    struct drag_result {
        std::optional<ValueType> start_value;
        bool changed;
    };

    template<draggable_type ValueType>
    auto infinite_drag(std::string_view name, ValueType& v, float speed = 1.0f, ValueType min = std::numeric_limits<ValueType>::lowest(), ValueType max = std::numeric_limits<ValueType>::max()) -> drag_result<ValueType> {
        static bool dragging = false;
        static ImVec2 drag_position = {};
        static ValueType drag_start_value = {};
        static ImGuiID dragging_id = 0;

        auto iv = v;
        bool changed = ImGui::DragScalar((std::string{"##"} + name.data()).c_str(), detail::get_data_type_of<ValueType>(), &v, speed, &min, &max);
        auto current_id = ImGui::GetID(name.data());
        bool hovered = ImGui::IsItemHovered();

        ImGui::SameLine();
        ImGui::TextUnformatted(name.data());

        if(dragging && current_id != dragging_id) return {std::nullopt, changed};

        bool mouse_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);

        if(!mouse_down && dragging) {
            dragging = false;
            dragging_id = 0;

            imftw::sig::set_cursor_position(drag_position.x, drag_position.y);
            imftw::sig::set_cursor_enabled(true);

            return {drag_start_value, changed};

        } else if(mouse_down && hovered && !dragging) {
            dragging = true;
            dragging_id = current_id;
            drag_position = ImGui::GetMousePos();
            drag_start_value = iv;

            imftw::sig::set_cursor_enabled(false);
        }

        return {std::nullopt, changed};
    }

    namespace scope {

        class [[nodiscard]] style {
        public:

            style() = delete;
            style(const style&) = delete;
            style(style&&) = delete;
            style& operator=(const style&) = delete;
            style& operator=(style&&) = delete;

            template<typename... Args>
            style(Args&&... args) {
                static_assert(sizeof...(Args) % 2 == 0, "style::style() requires an even number of arguments");
                static_assert(sizeof...(Args) > 1, "style::style() requires at least two arguments");

                push(std::forward<Args>(args)...);
            }

            ~style() {
                if (n_colors > 0) {
                    ImGui::PopStyleColor(n_colors);
                }

                if (n_styles > 0) {
                    ImGui::PopStyleVar(n_styles);
                }
            }
        private:

            template<typename StyleType, typename StyleValue, typename... Rest>
            void push(StyleType&& style_type, StyleValue&& style_value, Rest&&... rest) {
                push(std::forward<StyleType>(style_type), std::forward<StyleValue>(style_value));
                push(std::forward<Rest>(rest)...);
            }

            template<>
            void push<ImGuiCol_, ImVec4>(ImGuiCol_&& style_type, ImVec4&& style_value) {
                ImGui::PushStyleColor(std::forward<ImGuiCol_>(style_type), std::forward<ImVec4>(style_value));
                ++n_colors;
            }

            template<>
            void push<ImGuiCol_, ImU32>(ImGuiCol_&& style_type, ImU32&& style_value) {
                ImGui::PushStyleColor(std::forward<ImGuiCol_>(style_type), std::forward<ImU32>(style_value));
                ++n_colors;
            }

            constexpr static bool is_float_style_var(ImGuiStyleVar_ v) {
                return v == ImGuiStyleVar_Alpha
                    || v == ImGuiStyleVar_DisabledAlpha
                    || v == ImGuiStyleVar_WindowRounding
                    || v == ImGuiStyleVar_WindowBorderSize
                    || v == ImGuiStyleVar_ChildRounding
                    || v == ImGuiStyleVar_ChildBorderSize
                    || v == ImGuiStyleVar_PopupRounding
                    || v == ImGuiStyleVar_PopupBorderSize
                    || v == ImGuiStyleVar_FrameRounding
                    || v == ImGuiStyleVar_FrameBorderSize
                    || v == ImGuiStyleVar_IndentSpacing
                    || v == ImGuiStyleVar_ScrollbarSize
                    || v == ImGuiStyleVar_ScrollbarRounding
                    || v == ImGuiStyleVar_GrabMinSize
                    || v == ImGuiStyleVar_GrabRounding
                    || v == ImGuiStyleVar_TabRounding;
            }

            constexpr static bool is_vec2_style_var(ImGuiStyleVar_ v) {
                return v == ImGuiStyleVar_WindowPadding
                    || v == ImGuiStyleVar_WindowMinSize
                    || v == ImGuiStyleVar_WindowTitleAlign
                    || v == ImGuiStyleVar_FramePadding
                    || v == ImGuiStyleVar_ItemSpacing
                    || v == ImGuiStyleVar_ItemInnerSpacing
                    || v == ImGuiStyleVar_CellPadding
                    || v == ImGuiStyleVar_ButtonTextAlign
                    || v == ImGuiStyleVar_SeparatorTextAlign;
            }

            int n_colors = 0;
            int n_styles = 0;
        };

        class [[nodiscard]] id {
        public:
            id() = delete;
            id(const id&) = delete;
            id(id&&) = delete;
            id& operator=(const id&) = delete;
            id& operator=(id&&) = delete;

            explicit id(std::string_view id) {
                ImGui::PushID(id.data());
            }

            explicit id(const char* id) {
                ImGui::PushID(id);
            }

            explicit id(const void* id) {
                ImGui::PushID(id);
            }

            explicit id(int id) {
                ImGui::PushID(id);
            }

            ~id() {
                ImGui::PopID();
            }
        };

        class [[nodiscard]] enabled {
        public:
            enabled() = delete;
			enabled(const enabled&) = delete;
			enabled(enabled&&) = delete;
			enabled& operator=(const enabled&) = delete;
			enabled& operator=(enabled&&) = delete;

			enabled(bool enabled, float alpha = 0.5): pop(!enabled) {
                if (!enabled) {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * alpha);
                }
			}

			~enabled() {
                if (pop) {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleVar();
                }
			}

        private:
            bool pop = false;
		};
    }

    class [[nodiscard]] window {
    public:

        window() = delete;
        window(const window&) = delete;
        window(window&&) = delete;
        window& operator=(const window&) = delete;
        window& operator=(window&&) = delete;

        window(std::string_view title, ImGuiWindowFlags flags = 0, bool* open = nullptr) {
            is_open = ImGui::Begin(title.data(), open, flags);
        }

        ~window() {
            ImGui::End();
        }

        constexpr explicit(false) operator bool() const {
            return is_open;
        }

    private:
        bool is_open = false;
    };

    class [[nodiscard]] main_menu {
    public:

        main_menu(const main_menu&) = delete;
        main_menu(main_menu&&) = delete;
        main_menu& operator=(const main_menu&) = delete;
        main_menu& operator=(main_menu&&) = delete;

        main_menu() {
            open = ImGui::BeginMainMenuBar();
        }

        ~main_menu() {
            if (open) ImGui::EndMainMenuBar();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open = false;
    };

    class [[nodiscard]] menu_bar {
    public:

        menu_bar(const menu_bar&) = delete;
        menu_bar(menu_bar&&) = delete;
        menu_bar& operator=(const menu_bar&) = delete;
        menu_bar& operator=(menu_bar&&) = delete;

        menu_bar() {
            open = ImGui::BeginMenuBar();
        }

        ~menu_bar() {
            if (open) ImGui::EndMenuBar();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open;
    };

    class [[nodiscard]] menu {
    public:

        menu() = delete;
        menu(const menu&) = delete;
        menu(menu&&) = delete;
        menu& operator=(const menu&) = delete;
        menu& operator=(menu&&) = delete;

        menu(std::string_view title, bool enabled = true) {
            open = ImGui::BeginMenu(title.data(), enabled);
        }

        ~menu() {
            if (open) ImGui::EndMenu();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open;
    };

    class [[nodiscard]] popup {
    public:

        popup() = delete;
        popup(const popup&) = delete;
        popup(popup&&) = delete;
        popup& operator=(const popup&) = delete;
        popup& operator=(popup&&) = delete;

        popup(std::string_view title, bool modal = false, ImGuiWindowFlags flags = 0) {
            open = modal ? ImGui::BeginPopupModal(title.data(), nullptr, flags) : ImGui::BeginPopup(title.data(), flags);
        }

        ~popup() {
            if (open) ImGui::EndPopup();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open = false;
    };

}

#ifndef IMFTW_NO_SHORTCUTS
    #define IMFTW_WITH_STYLE(...) if(imftw::scope::style _imftw_style(__VA_ARGS__); true)
    #define IMFTW_WITH_ID(...) if(imftw::scope::id _imftw_id(__VA_ARGS__); true)
    #define IMFTW_WITH_ENABLED(...) if(imftw::scope::enabled _imftw_enabled(__VA_ARGS__); true)
    #define IMFTW_POPUP(...) if(imftw::popup _imftw_popup(__VA_ARGS__); _imftw_popup)
    #define IMFTW_MENU(...) if(imftw::menu _imftw_menu(__VA_ARGS__); _imftw_menu)
    #define IMFTW_MAIN_MENU() if(imftw::main_menu _imftw_main_menu; _imftw_main_menu)
    #define IMFTW_MENU_BAR() if(imftw::menu_bar _imftw_menu_bar; _imftw_menu_bar)
    #define IMFTW_WINDOW(...) if(imftw::window _imftw_window(__VA_ARGS__); _imftw_window)
#endif
