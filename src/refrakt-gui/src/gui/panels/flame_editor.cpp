#include <imftw/imftw.h>
#include <imftw/gui.h>

#include <IconsMaterialDesign.h>

#include <librefrakt/flame_info.h>

#include "gui/panels/flame_editor.h"

command_executor::command_t make_undoer(rfkt::flame& f, const rfkt::accessor& desc, rfkt::anima&& new_value, rfkt::anima&& old_value) {
	return {
		[&f, desc, new_value = std::move(new_value)]() mutable {
			*desc.access(f) = new_value;
		},
		[&f, desc, old_value = std::move(old_value)]() mutable {
			*desc.access(f) = old_value;
		},
		std::format("Modify {}", desc.to_string())
	};
}

auto make_animator_button(rfkt::flame& f, const rfkt::function_table& ft, command_executor& exec) {
	return [&ft, &exec, &f](const rfkt::accessor& desc, float width) -> bool {

		auto& ani = *desc.access(f);

		if (ImGui::Button(ani.call_info ? ICON_MD_ANIMATION : "", { width , 0 })) {
			ImGui::OpenPopup("animation_editor");
		}

		if (!ImGui::BeginPopup("animation_editor")) { return false; }

		bool value_changed = false;


		std::string cur_selected = ani.call_info ? ani.call_info->name : "none";
		std::optional<std::string_view> selection = std::nullopt;

		if (ImGui::BeginCombo("##a_type", cur_selected.c_str())) {
			if (ImGui::Selectable("none")) selection = "none";
			if (!ani.call_info) ImGui::SetItemDefaultFocus();

			for (auto& name : ft.names()) {
				bool is_selected = cur_selected == name;
				if (ImGui::Selectable(name.c_str(), &is_selected))
					selection = name;
				if (is_selected) ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		if (selection.has_value() && selection != cur_selected) {
			value_changed = true;
			rfkt::anima::call_info_t new_call_info = selection == "none" ? std::nullopt : ft.make_default(*selection);
			rfkt::anima::call_info_t old_call_info = ani.call_info;

			exec.execute(make_undoer(f, desc, { ani.t0, std::move(new_call_info) }, { ani.t0, std::move(old_call_info) }));
		}

		if (ani.call_info) {
			bool changed = false;
			rfkt::anima::call_info_t old_call_info = ani.call_info;
			for (auto& [name, arg] : ani.call_info->args) {
				changed |= std::visit([&name]<typename T>(T & argv) {
					if constexpr (std::same_as<T, double>) {
						return ImGui::InputDouble(name.c_str(), &argv);
					}
					else if constexpr (std::same_as<T, int>) {
						return ImGui::InputInt(name.c_str(), &argv);
					}
					else if constexpr (std::same_as<T, bool>) {
						return ImGui::Checkbox(name.c_str(), &argv);
					}
					else if constexpr (std::same_as<T, std::string>) {
						return false;
					}
					else {
						SPDLOG_ERROR("Unhandled case: {}", typeid(T).name());
					}
				}, arg);
			}

			if (changed) {
				rfkt::anima::call_info_t new_call_info = ani.call_info;
				exec.execute(make_undoer(f, desc, { ani.t0, std::move(new_call_info) }, { ani.t0, std::move(old_call_info) }));
				value_changed = true;
			}
		}


		ImGui::EndPopup();
		return value_changed;
	};
}

struct edit_bounds {
	double min = -DBL_MAX;
	double max = DBL_MAX;

	edit_bounds() = default;
	explicit(false) edit_bounds(double min) : min{ min } {}
	edit_bounds(double min, double max) : min{ min }, max{ max } {}
};

auto make_flame_drag_edit(rfkt::flame& f, command_executor& exec, const rfkt::function_table& ft, bool& value_changed) {
	auto min_button_width = ImGui::CalcTextSize(ICON_MD_ANIMATION).x + ImGui::GetStyle().FramePadding.x * 2.0f;
	auto animator_button = make_animator_button(f, ft, exec);
	return[&f, &exec, animator_button = std::move(animator_button), min_button_width, &value_changed](const rfkt::accessor& desc, std::string_view text, float step, const edit_bounds& eb = {}) mutable {
		ImFtw::Scope::ID drag_scope{ desc.hash().str16() };

		auto* ptr = desc.access(f);
		value_changed |= animator_button(desc, min_button_width);
		ImGui::SameLine();
		if (auto [drag_start, changed] = ImFtw::InfiniteDrag(text, ptr->t0, step, eb.min, eb.max); drag_start) {
			value_changed = true;
			rfkt::anima new_value = { ptr->t0, ptr->call_info };
			rfkt::anima old_value = { drag_start.value(), ptr->call_info };
			exec.execute(make_undoer(f, desc, std::move(new_value), std::move(old_value)));
		}

	};
}


bool rfkt::gui::panels::flame_editor(rfkt::flamedb& fdb, rfkt::flame& f, command_executor& exec, rfkt::function_table& ft) {
	bool changed = false;

	namespace fdesc = rfkt::accessors;

	ImFtw::Scope::ID flame_scope{ &f };

	auto flame_drag_edit = make_flame_drag_edit(f, exec, ft, changed);

	ImGui::Text("Display");
	flame_drag_edit(fdesc::flame{ &rfkt::flame::rotate }, "Rotate", 0.01f);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::scale }, "Scale", 0.1f, 0.0001);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::center_x }, "Center X", 0.1f);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::center_y }, "Center Y", 0.1f);
	ImGui::Separator();

	ImGui::Text("Color");
	flame_drag_edit(fdesc::flame{ &rfkt::flame::gamma }, "Gamma", 0.01f, 0.1);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::brightness }, "Brightness", 0.01f, 0.1);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::vibrancy }, "Vibrancy", 0.01f);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::mod_hue}, "Mod Hue", 0.01f);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::mod_sat}, "Mod Sat", 0.01f);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::mod_val}, "Mod Val", 0.01f);
	ImGui::Separator();

	auto invoker = [&ft]<typename... Args>(Args&&... args) { return ft.call(std::forward<Args>(args)...); };

	static bool hide_linear = false;
	ImGui::Checkbox("Hide Linear Only XForms", &hide_linear);
	ImGui::SameLine();

	ImFtw::Tooltip("Hide XForms that only contain a single linear VLink");

	f.for_each_xform([&](int xid, rfkt::xform& xf) {
		ImFtw::Scope::ID xf_scope{ xid };

		std::string xf_name = (xid == -1) ? "Final XForm" : fmt::format("XForm {}", xid);

		bool linear_only = (xid != -1 && xf.vchain.size() == 1 && xf.vchain[0].size_variations() == 1 && xf.vchain[0].has_variation("linear"));

		if (linear_only && hide_linear)
			return;

		if (ImGui::CollapsingHeader(xf_name.c_str())) {
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::weight }, "Weight", 0.01, 0);
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::color }, "Color", 0.001, { 0, 1 });
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::color_speed }, "Color Speed", 0.001, { 0, 1 });
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::opacity }, "Opacity", 0.001, { 0, 1 });

			if (ImGui::BeginTabBar("VChain")) {
				for (int i = 0; i < xf.vchain.size(); i++) {
					if (ImGui::BeginTabItem(std::format("VL {}", i + 1).c_str())) {
						//ImGui::BeginChild(std::format("vchain{}", i).c_str(), ImVec2(0, 200));

						auto& vl = xf.vchain[i];
						{
							ImFtw::Scope::ID vl_scope{ i };

							ImGui::Text("Affine");
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::a }, "A", 0.001f);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::b }, "B", 0.001f);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::c }, "C", 0.001f);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::d }, "D", 0.001f);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::e }, "E", 0.001f);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::f }, "F", 0.001f);
							flame_drag_edit(fdesc::vlink{ xid, i, &rfkt::vlink::mod_rotate }, "Rotate", 0.01f);
							flame_drag_edit(fdesc::vlink{ xid, i, &rfkt::vlink::mod_scale }, "Scale", 0.01f, 0.0001);
							flame_drag_edit(fdesc::vlink{ xid, i, &rfkt::vlink::mod_x }, "X", 0.01f);
							flame_drag_edit(fdesc::vlink{ xid, i, &rfkt::vlink::mod_y }, "Y", 0.01f);
							ImGui::Separator();

							//static bool just_opened = false;
							static char filter[128] = { 0 };

							if (ImGui::Button("Add variation")) {
								ImGui::OpenPopup("add_variation");
								//just_opened = true;
								std::memset(filter, 0, sizeof(filter));
							}

							if (ImGui::BeginPopup("add_variation")) {
								//if (just_opened) {
								//	just_opened = false;
								//	ImGui::SetKeyboardFocusHere(1);
								//}

								if (ImGui::IsWindowAppearing()) {
									ImGui::SetKeyboardFocusHere();
								}

								bool enter_pressed = ImGui::InputText("##filter", filter, sizeof(filter), ImGuiInputTextFlags_EnterReturnsTrue);
								std::string_view selected_var{};

								if (ImGui::BeginListBox("##vlist", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing()))) {
									for (const auto& var : fdb.variations()) {
										if (!vl.has_variation(var.name) && (strlen(filter) < 2 || var.name.find(filter) != std::string::npos)) {
											if (ImGui::Selectable(var.name.c_str()) || (enter_pressed && fdb.is_variation(filter))) {
												selected_var = var.name;
												ImGui::CloseCurrentPopup();
											}
										}
									}
									ImGui::EndListBox();
								}
								ImGui::EndPopup();

								if (!selected_var.empty()) {
									changed = true;
									exec.execute(
										[&f, xid, i, &fdb, name = std::string{ selected_var }]() {
											f.get_xform(xid)->vchain[i].add_variation(fdb.make_vardata(name));
										},
										[&f, xid, i, name = std::string{ selected_var }]() {
											f.get_xform(xid)->vchain[i].remove_variation(name);
										}
										);
								}
							}

							std::string_view removed_var = {};
							for (auto& [vname, vd] : vl) {
								ImFtw::Scope::ID var_scope{ vname };
								ImGui::Separator();
								flame_drag_edit(fdesc::vardata{ xid, i, vname, &rfkt::vardata::weight }, vname, 0.001);

								if (vl.size_variations() > 1) {
									ImGui::SameLine(ImGui::GetContentRegionAvail().x - 10);
									if (ImGui::Button("X")) removed_var = vname;
								}

								for (auto& [pname, val] : vd) {
									std::string full_name = std::format("{}_{}", vname, pname);
									flame_drag_edit(fdesc::parameter{ xid, i, vname, pname }, full_name, 0.001);
								}
							}
							if (!removed_var.empty()) {
								changed = true;
								exec.execute(
									[&f, xid, i, name = std::string{ removed_var }]() {
										f.get_xform(xid)->vchain[i].remove_variation(name);
									},
									[&f, xid, i, name = std::string{ removed_var }, vdata = vl[removed_var]] {
										f.get_xform(xid)->vchain[i].add_variation({ name, vdata });
									}
									);
							}
						}
						//ImGui::EndChild();
						ImGui::EndTabItem();
					}
				}
				ImGui::EndTabBar();
			}
		}
	});

	return changed;
}