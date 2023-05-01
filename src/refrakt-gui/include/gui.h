#pragma once

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <imgui_internal.h>
#include <implot.h>

inline ImVec2 operator+(const ImVec2& l, const ImVec2& r) {
	return { l.x + r.x, l.y + r.y };
}

inline ImVec2 operator-(const ImVec2& l, const ImVec2& r) {
	return { l.x - r.x, l.y - r.y };
}

namespace rfkt::gui {

	struct style_info {
		std::string name;
		std::string author;
		std::string url;
		std::size_t id;
	};

	std::vector<style_info> get_styles();
	void set_style(std::size_t id);

	template<typename T>
	struct [[nodiscard]] id_scope {
		id_scope(T id) { ImGui::PushID(id); }
		~id_scope() { ImGui::PopID(); }
	};

	std::optional<double> drag_double(std::string_view name, double& v, float speed, double min, double max);

	inline void tooltip(const std::string& tip) {
		ImGui::TextDisabled("(?)");
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(tip.c_str());
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

	template<typename... Args>
	void tooltip(const std::string& tip, Args&&... args) {
		tooltip(fmt::format(tip, args...));
	}

	namespace scope {

		template<typename... Args>
		class [[nodiscard]] style {
		public:

			style() = delete;

			style(const style&) = delete;
			style& operator=(const style&) = delete;

			style(style&&) = delete;
			style& operator=(style&&) = delete;

			style(Args&&... args) {
				static_assert(sizeof...(args) % 2 == 0, "style_scope requires even arguments");
				push(args...);
			}

			~style() {
				ImGui::PopStyleVar(nargs);
			}

		private:

			template<typename V, typename T, typename... Tail>
			void push(const V& idx, const T& v, Tail&&... tail) {
				ImGui::PushStyleVar(idx, v);
				push(tail...);
			}

			template<typename V, typename T>
			void push(const V& idx, const T& v) {
				ImGui::PushStyleVar(idx, v);
			}

			static constexpr std::size_t nargs = sizeof...(Args) / 2;


		};

		class [[nodiscard]] window {
		public:
			window(const std::string& name, bool* open = nullptr, ImGuiWindowFlags flags = 0) {
				ImGui::Begin(name.c_str(), open, flags);
			}

			~window() {
				ImGui::End();
			}
		};

		class [[nodiscard]] enabled {
		public:
			enabled(bool enabled, float alpha) : pop(!enabled) {
				if (!enabled) {
					ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
					ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * alpha);
				}
			}

			~enabled() {
				if (pop) {
					ImGui::PopStyleVar();
					ImGui::PopItemFlag();
				}
			}

		private:
			bool pop = false;
		};
	}
}