#pragma once

#include <imgui.h>
#include <imgui_internal.h>

#include <imftw/signals.h>

namespace ImFtw {

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
                static_assert("Unsupported type for ImFtw::detail::get_data_type_of()");
            }
        }


    }

    inline void Tooltip(std::string_view text, bool show_icon = true) {
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

    inline void TextCentered(std::string_view text) {
        auto region = ImGui::GetContentRegionAvail();
        auto size = ImGui::CalcTextSize(text.data());
        ImGui::SetCursorPosX(region.x / 2 - size.x / 2);
        ImGui::TextUnformatted(text.data());
    }

    template<typename T>
    concept DraggableType = 
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

    template<DraggableType ValueType>
    struct DragResult {
        // If a dragging action was just finished, this will contain the value
        // before the drag started.
        std::optional<ValueType> StartValue;

        bool Changed;
    };

    // Like the regular ImGui::DragScalar, but allows for infinite dragging.
    // The mouse cursor will be hidden while dragging and will not collide
    // with screen edges. The cursor will be reset to its original position
    // when the drag is finished.
    template<DraggableType ValueType>
    auto InfiniteDrag(std::string_view name, ValueType& v, float speed = 1.0f, ValueType min = std::numeric_limits<ValueType>::lowest(), ValueType max = std::numeric_limits<ValueType>::max()) -> DragResult<ValueType> {
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

        if (dragging) {
            ImGui::SetKeyOwner(ImGuiKey_MouseWheelX, dragging_id);
            ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, dragging_id);
        }

        if(!mouse_down && dragging) {
            dragging = false;
            dragging_id = 0;

            ImFtw::Sig::SetCursorPosition(drag_position.x, drag_position.y);
            ImFtw::Sig::SetCursorEnabled(true);

            return {drag_start_value, changed};

        } else if(mouse_down && hovered && !dragging) {
            dragging = true;
            dragging_id = current_id;
            drag_position = ImGui::GetMousePos();
            drag_start_value = iv;

            ImFtw::Sig::SetCursorEnabled(false);
        }
        else if (auto scroll = ImGui::GetIO().MouseWheel; dragging && scroll != 0.0) {
            v = std::min(max, std::max(min, static_cast<ValueType>(v + scroll * speed)));
            if(v != iv) changed = true;
        }

        return {std::nullopt, changed};
    }

    namespace Scope {

        // RAII helper for ImGui styling.
        class [[nodiscard]] Style {
        public:

            Style() = delete;
            Style(const Style&) = delete;
            Style(Style&&) = delete;
            Style& operator=(const Style&) = delete;
            Style& operator=(Style&&) = delete;

            template<typename... Args>
            Style(Args&&... args) {
                static_assert(sizeof...(Args) % 2 == 0, "Style::Style() requires an even number of arguments");
                static_assert(sizeof...(Args) > 1, "Style::Style() requires at least two arguments");

                push(std::forward<Args>(args)...);
            }

            ~Style() {
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

        // RAII helper for ImGui's ID stack.
        class [[nodiscard]] ID {
        public:
            ID() = delete;
            ID(const ID&) = delete;
            ID(ID&&) = delete;
            ID& operator=(const ID&) = delete;
            ID& operator=(ID&&) = delete;

            explicit ID(std::string_view id) {
                ImGui::PushID(id.data());
            }

            explicit ID(const char* id) {
                ImGui::PushID(id);
            }

            explicit ID(const void* id) {
                ImGui::PushID(id);
            }

            explicit ID(int id) {
                ImGui::PushID(id);
            }

            ~ID() {
                ImGui::PopID();
            }
        };

        // RAII helper for enabling/disabling ImGui items.
        class [[nodiscard]] Enabled {
        public:
            Enabled() = delete;
            Enabled(const Enabled&) = delete;
            Enabled(Enabled&&) = delete;
            Enabled operator=(const Enabled) = delete;
            Enabled& operator=(Enabled&&) = delete;

            Enabled(bool enabled, float alpha = 0.5): pop(!enabled) {
                if (!enabled) {
                    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * alpha);
                }
			}

			~Enabled() {
                if (pop) {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleVar();
                }
			}

        private:
            bool pop = false;
		};
    }

    class [[nodiscard]] Window {
    public:

        Window() = delete;
        Window(const Window&) = delete;
        Window(Window&&) = delete;
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&&) = delete;

        Window(std::string_view title, ImGuiWindowFlags flags = 0, bool* open = nullptr) {
            is_open = ImGui::Begin(title.data(), open, flags);
        }

        ~Window() {
            ImGui::End();
        }

        constexpr explicit(false) operator bool() const {
            return is_open;
        }

    private:
        bool is_open = false;
    };

    class [[nodiscard]] MainMenu {
    public:

        MainMenu(const MainMenu&) = delete;
        MainMenu(MainMenu&&) = delete;
        MainMenu& operator=(const MainMenu&) = delete;
        MainMenu& operator=(MainMenu&&) = delete;

        MainMenu() {
            open = ImGui::BeginMainMenuBar();
        }

        ~MainMenu() {
            if (open) ImGui::EndMainMenuBar();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open = false;
    };

    class [[nodiscard]] MenuBar {
    public:

        MenuBar(const MenuBar&) = delete;
        MenuBar(MenuBar&&) = delete;
        MenuBar& operator=(const MenuBar&) = delete;
        MenuBar& operator=(MenuBar&&) = delete;

        MenuBar() {
            open = ImGui::BeginMenuBar();
        }

        ~MenuBar() {
            if (open) ImGui::EndMenuBar();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open;
    };

    class [[nodiscard]] Menu {
    public:

        Menu() = delete;
        Menu(const Menu&) = delete;
        Menu(Menu&&) = delete;
        Menu& operator=(const Menu&) = delete;
        Menu& operator=(Menu&&) = delete;

        Menu(std::string_view title, bool enabled = true) {
            open = ImGui::BeginMenu(title.data(), enabled);
        }

        ~Menu() {
            if (open) ImGui::EndMenu();
        }

        constexpr explicit(false) operator bool() const noexcept {
            return open;
        }

    private:
        bool open;
    };

    class [[nodiscard]] Popup {
    public:

        Popup() = delete;
        Popup(const Popup&) = delete;
        Popup(Popup&&) = delete;
        Popup& operator=(const Popup&) = delete;
        Popup& operator=(Popup&&) = delete;

        Popup(std::string_view title, bool modal = false, ImGuiWindowFlags flags = 0) {
            open = modal ? ImGui::BeginPopupModal(title.data(), nullptr, flags) : ImGui::BeginPopup(title.data(), flags);
        }

        ~Popup() {
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
    #define IMFTW_WITH_STYLE(...) if(ImFtw::Scope::Style _imftw_style(__VA_ARGS__); true)
    #define IMFTW_WITH_ID(...) if(ImFtw::Scope::ID _imftw_id(__VA_ARGS__); true)
    #define IMFTW_WITH_ENABLED(...) if(ImFtw::Scope::Enabled _imftw_enabled(__VA_ARGS__); true)
    #define IMFTW_POPUP(...) if(ImFtw::Popup _imftw_popup(__VA_ARGS__); _imftw_popup)
    #define IMFTW_MENU(...) if(ImFtw::Menu _imftw_menu(__VA_ARGS__); _imftw_menu)
    #define IMFTW_MAIN_MENU() if(ImFtw::MainMenu _imftw_main_menu; _imftw_main_menu)
    #define IMFTW_MENU_BAR() if(ImFtw::MenuBar _imftw_menu_bar; _imftw_menu_bar)
    #define IMFTW_WINDOW(...) if(ImFtw::Window _imftw_window(__VA_ARGS__); _imftw_window)
#endif
