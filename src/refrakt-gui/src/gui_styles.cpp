#include <imgui.h>
#include <yaml-cpp/yaml.h>

#include "gl.h"
#include "gui.h"

YAML::Node& styles() {
	static YAML::Node styles_ = YAML::LoadFile("config/styles.yml");
	return styles_;
}
const std::map<std::string, ImGuiCol>& color_map() {
	static std::map<std::string, ImGuiCol> color_map_ = {
		{"Text", ImGuiCol_Text},
		{"TextDisabled", ImGuiCol_TextDisabled},
		{"WindowBg", ImGuiCol_WindowBg},
		{"ChildBg", ImGuiCol_ChildBg},
		{"PopupBg", ImGuiCol_PopupBg},
		{"Border", ImGuiCol_Border},
		{"BorderShadow", ImGuiCol_BorderShadow},
		{"FrameBg", ImGuiCol_FrameBg},
		{"FrameBgHovered", ImGuiCol_FrameBgHovered},
		{"FrameBgActive", ImGuiCol_FrameBgActive},
		{"TitleBg", ImGuiCol_TitleBg},
		{"TitleBgActive", ImGuiCol_TitleBgActive},
		{"TitleBgCollapsed", ImGuiCol_TitleBgCollapsed},
		{"MenuBarBg", ImGuiCol_MenuBarBg},
		{"ScrollbarBg", ImGuiCol_ScrollbarBg},
		{"ScrollbarGrab", ImGuiCol_ScrollbarGrab},
		{"ScrollbarGrabHovered", ImGuiCol_ScrollbarGrabHovered},
		{"ScrollbarGrabActive", ImGuiCol_ScrollbarGrabActive},
		{"CheckMark", ImGuiCol_CheckMark},
		{"SliderGrab", ImGuiCol_SliderGrab},
		{"SliderGrabActive", ImGuiCol_SliderGrabActive},
		{"Button", ImGuiCol_Button},
		{"ButtonHovered", ImGuiCol_ButtonHovered},
		{"ButtonActive", ImGuiCol_ButtonActive},
		{"Header", ImGuiCol_Header},
		{"HeaderHovered", ImGuiCol_HeaderHovered},
		{"HeaderActive", ImGuiCol_HeaderActive},
		{"Separator", ImGuiCol_Separator},
		{"SeparatorHovered", ImGuiCol_SeparatorHovered},
		{"SeparatorActive", ImGuiCol_SeparatorActive},
		{"ResizeGrip", ImGuiCol_ResizeGrip},
		{"ResizeGripHovered", ImGuiCol_ResizeGripHovered},
		{"ResizeGripActive", ImGuiCol_ResizeGripActive},
		{"Tab", ImGuiCol_Tab},
		{"TabHovered", ImGuiCol_TabHovered},
		{"TabActive", ImGuiCol_TabActive},
		{"TabUnfocused", ImGuiCol_TabUnfocused},
		{"TabUnfocusedActive", ImGuiCol_TabUnfocusedActive},
		{"DockingPreview", ImGuiCol_DockingPreview},
		{"DockingEmptyBg", ImGuiCol_DockingEmptyBg},
		{"PlotLines", ImGuiCol_PlotLines},
		{"PlotLinesHovered", ImGuiCol_PlotLinesHovered},
		{"PlotHistogram", ImGuiCol_PlotHistogram},
		{"PlotHistogramHovered", ImGuiCol_PlotHistogramHovered},
		{"TableHeaderBg", ImGuiCol_TableHeaderBg},
		{"TableBorderStrong", ImGuiCol_TableBorderStrong},
		{"TableBorderLight", ImGuiCol_TableBorderLight},
		{"TableRowBg", ImGuiCol_TableRowBg},
		{"TableRowBgAlt", ImGuiCol_TableRowBgAlt},
		{"TextSelectedBg", ImGuiCol_TextSelectedBg},
		{"DragDropTarget", ImGuiCol_DragDropTarget},
		{"NavHighlight", ImGuiCol_NavHighlight},
		{"NavWindowingHighlight", ImGuiCol_NavWindowingHighlight},
		{"NavWindowingDimBg", ImGuiCol_NavWindowingDimBg},
		{"ModalWindowDimBg", ImGuiCol_ModalWindowDimBg}
	};

	return color_map_;
}

enum class ImGuiConfigType {
	float32,
	vec,
	integer,
	boolean
};

const std::map<std::string, std::pair<unsigned long long, ImGuiConfigType>>& config_map() {
	static std::map<std::string, std::pair<unsigned long long, ImGuiConfigType>> config_map_ = {
		{ "Alpha", {offsetof(ImGuiStyle, Alpha), ImGuiConfigType::float32} },
		{ "DisabledAlpha", {offsetof(ImGuiStyle, DisabledAlpha), ImGuiConfigType::float32} },
		{ "WindowPadding", {offsetof(ImGuiStyle, WindowPadding), ImGuiConfigType::vec} },
		{ "WindowRounding", {offsetof(ImGuiStyle, WindowRounding), ImGuiConfigType::float32} },
		{ "WindowBorderSize", {offsetof(ImGuiStyle, WindowBorderSize), ImGuiConfigType::float32} },
		{ "WindowMinSize", {offsetof(ImGuiStyle, WindowMinSize), ImGuiConfigType::vec} },
		{ "WindowTitleAlign", {offsetof(ImGuiStyle, WindowTitleAlign), ImGuiConfigType::vec} },
		{ "WindowMenuButtonPosition", {offsetof(ImGuiStyle, WindowMenuButtonPosition), ImGuiConfigType::integer} },
		{ "ChildRounding", {offsetof(ImGuiStyle, ChildRounding), ImGuiConfigType::float32} },
		{ "ChildBorderSize", {offsetof(ImGuiStyle, ChildBorderSize), ImGuiConfigType::float32} },
		{ "PopupRounding", {offsetof(ImGuiStyle, PopupRounding), ImGuiConfigType::float32} },
		{ "PopupBorderSize", {offsetof(ImGuiStyle, PopupBorderSize), ImGuiConfigType::float32} },
		{ "FramePadding", {offsetof(ImGuiStyle, FramePadding), ImGuiConfigType::vec} },
		{ "FrameRounding", {offsetof(ImGuiStyle, FrameRounding), ImGuiConfigType::float32} },
		{ "FrameBorderSize", {offsetof(ImGuiStyle, FrameBorderSize), ImGuiConfigType::float32} },
		{ "ItemSpacing", {offsetof(ImGuiStyle, ItemSpacing), ImGuiConfigType::vec} },
		{ "ItemInnerSpacing", {offsetof(ImGuiStyle, ItemInnerSpacing), ImGuiConfigType::vec} },
		{ "CellPadding", {offsetof(ImGuiStyle, CellPadding), ImGuiConfigType::vec} },
		{ "TouchExtraPadding", {offsetof(ImGuiStyle, TouchExtraPadding), ImGuiConfigType::vec} },
		{ "IndentSpacing", {offsetof(ImGuiStyle, IndentSpacing), ImGuiConfigType::float32} },
		{ "ColumnsMinSpacing", {offsetof(ImGuiStyle, ColumnsMinSpacing), ImGuiConfigType::float32} },
		{ "ScrollbarSize", {offsetof(ImGuiStyle, ScrollbarSize), ImGuiConfigType::float32} },
		{ "ScrollbarRounding", {offsetof(ImGuiStyle, ScrollbarRounding), ImGuiConfigType::float32} },
		{ "GrabMinSize", {offsetof(ImGuiStyle, GrabMinSize), ImGuiConfigType::float32} },
		{ "GrabRounding", {offsetof(ImGuiStyle, GrabRounding), ImGuiConfigType::float32} },
		{ "LogSliderDeadzone", {offsetof(ImGuiStyle, LogSliderDeadzone), ImGuiConfigType::float32} },
		{ "TabRounding", {offsetof(ImGuiStyle, TabRounding), ImGuiConfigType::float32} },
		{ "TabBorderSize", {offsetof(ImGuiStyle, TabBorderSize), ImGuiConfigType::float32} },
		{ "TabMinWidthForCloseButton", {offsetof(ImGuiStyle, TabMinWidthForCloseButton), ImGuiConfigType::float32} },
		{ "ColorButtonPosition", {offsetof(ImGuiStyle, ColorButtonPosition), ImGuiConfigType::integer} },
		{ "ButtonTextAlign", {offsetof(ImGuiStyle, ButtonTextAlign), ImGuiConfigType::vec} },
		{ "SelectableTextAlign", {offsetof(ImGuiStyle, SelectableTextAlign), ImGuiConfigType::vec} },
		{ "SeparatorTextBorderSize", {offsetof(ImGuiStyle, SeparatorTextBorderSize), ImGuiConfigType::float32} },
		{ "SeparatorTextAlign", {offsetof(ImGuiStyle, SeparatorTextAlign), ImGuiConfigType::vec} },
		{ "SeparatorTextPadding", {offsetof(ImGuiStyle, SeparatorTextPadding), ImGuiConfigType::vec} },
		{ "DisplayWindowPadding", {offsetof(ImGuiStyle, DisplayWindowPadding), ImGuiConfigType::vec} },
		{ "DisplaySafeAreaPadding", {offsetof(ImGuiStyle, DisplaySafeAreaPadding), ImGuiConfigType::vec} },
		{ "MouseCursorScale", {offsetof(ImGuiStyle, MouseCursorScale), ImGuiConfigType::float32} },
		{ "AntiAliasedLines", {offsetof(ImGuiStyle, AntiAliasedLines), ImGuiConfigType::boolean} },
		{ "AntiAliasedLinesUseTex", {offsetof(ImGuiStyle, AntiAliasedLinesUseTex), ImGuiConfigType::boolean} },
		{ "AntiAliasedFill", {offsetof(ImGuiStyle, AntiAliasedFill), ImGuiConfigType::boolean} },
		{ "CurveTessellationTol", {offsetof(ImGuiStyle, CurveTessellationTol), ImGuiConfigType::float32} },
		{ "CircleTessellationMaxError", {offsetof(ImGuiStyle, CircleTessellationMaxError), ImGuiConfigType::float32} }
	};

	return config_map_;
}

std::vector<rfkt::gui::style_info> rfkt::gui::get_styles() {

	std::vector<gui::style_info> ret{};
	std::size_t id = 0;
	for (auto s : styles()) {
		ret.push_back({
			s["name"].as<std::string>(),
			s["author"].as<std::string>(),
			s["url"].as<std::string>(),
			id++
			});
	}

	return ret;
}

void rfkt::gui::set_style(std::size_t idx) {
	if (idx > styles().size()) return;

	auto style_info = styles()[idx];
	auto& style = ImGui::GetStyle();
	auto& cols = color_map();
	auto& conf = config_map();

	for (auto c : style_info["colors"]) {
		auto c_name = c.first.as<std::string>();
		if (!cols.contains(c_name)) {
			continue;
		}
		style.Colors[cols.at(c_name)] = ImVec4(c.second[0].as<float>(), c.second[1].as<float>(), c.second[2].as<float>(), c.second[3].as<float>());
	}

	for (auto c : style_info["config"]) {
		auto c_name = c.first.as<std::string>();
		if (!conf.contains(c_name)) {
			continue;
		}
	
		auto&& [offset, type] = conf.at(c_name);
		void* c_pos = ((char*)&style) + offset;

		if (c.second.IsScalar()) {
			switch (type) {
			case ImGuiConfigType::boolean:
				*((bool*)c_pos) = c.second.as<bool>();
				break;
			case ImGuiConfigType::integer:
				*((int*)c_pos) = c.second.as<int>();
				break;
			case ImGuiConfigType::float32:
				*((float*)c_pos) = c.second.as<float>();
			}
		}
		else {
			*((ImVec2*)c_pos) = ImVec2(c.second[0].as<float>(), c.second[1].as<float>());
		}
	}
}

std::optional<double> rfkt::gui::drag_double(std::string_view name, double& v, float speed, double min, double max) {

	static bool dragging = false;
	static ImVec2 drag_position = {};
	static double drag_start = 0.0;
	static ImGuiID dragging_id = 0;

	auto iv = v;
	ImGui::DragScalar(std::format("##{}", name).c_str(), ImGuiDataType_Double, &v, speed, &min, &max);
	auto cur_id = ImGui::GetID(name.data());

	if (dragging && dragging_id != cur_id) return std::nullopt;

	const bool mouse_down = ImGui::IsMouseDown(ImGuiMouseButton_Left);
	const bool hovered = ImGui::IsItemHovered();

	ImGui::SameLine();
	ImGui::Text(name.data());

	if (!mouse_down && dragging) {
		dragging = false;

		/*exec(
			[&ref = v, v]() mutable { SPDLOG_INFO("redoing to {}", v);  ref = v; },
			[&ref = v, iv = drag_start]() mutable { SPDLOG_INFO("undoing to {}", iv); ref = iv; }
		);*/

		dragging_id = 0;
		rfkt::gl::set_mouse_position(drag_position.x, drag_position.y);
		rfkt::gl::set_cursor_enabled(true);

		return (v != drag_start) ? std::make_optional(drag_start) : std::nullopt;
	}
	else if (mouse_down && !dragging && hovered) {
		dragging = true;
		dragging_id = cur_id;
		drag_start = iv;
		rfkt::gl::set_cursor_enabled(false);
		drag_position = ImGui::GetMousePos();
	}

	return std::nullopt;
}
