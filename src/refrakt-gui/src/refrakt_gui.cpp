#include <cuda.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <concurrencpp/concurrencpp.h>
#include <RtMidi.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/image/tonemapper.h>
#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/converter.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/nvjpeg.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/gpuinfo.h>

#include <librefrakt/util/http.h>

#include "gl.h"
#include "gui.h"

#include "gui/modals/render_modal.h"
#include "gui/panels/preview_panel.h"

class command_executor {
public:

	command_executor() : current_command{ command_stack.end() } {}

	void execute(std::pair<thunk_t, thunk_t>&& funcs) {
		funcs.first();

		if (can_redo()) {
			command_stack.erase(current_command, command_stack.end());
		}

		command_stack.push_back({ std::move(funcs.second), std::move(funcs.first) });
		current_command = command_stack.end();
	}

	bool can_undo() {
		return current_command != command_stack.begin();
	}

	bool can_redo() {
		return current_command != command_stack.end();
	}

	void undo() {
		if (!can_undo())
			return;

		--current_command;
		current_command->undo();
	}

	void redo() {
		if (!can_redo())
			return;

		current_command->redo();
		++current_command;
	}

	void clear() {
		command_stack.clear();
		current_command = command_stack.end();
	}

private:

	struct undo_redo {
		thunk_t undo;
		thunk_t redo;
	};

	std::list<undo_redo> command_stack{};
	decltype(command_stack)::iterator current_command;
};

void draw_status_bar() {
	ImGuiViewportP* viewport = (ImGuiViewportP*)(void*)ImGui::GetMainViewport();
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	if (ImGui::BeginViewportSideBar("##StatusBar", viewport, ImGuiDir_Down, ImGui::GetFrameHeight(), window_flags)) {
		if (ImGui::BeginMenuBar()) {
			auto dev = rfkt::gpuinfo::device::by_index(0);
			auto temp = dev.temperature();

			auto temp_color = [&]() {

				constexpr auto coolest = 30.0;
				constexpr auto hottest = 80.0;

				if (temp <= coolest) return IM_COL32(0, 255, 0, 255);
				if (temp >= hottest) return IM_COL32(255, 0, 0, 255);

				auto mix = (temp - coolest) / (hottest - coolest);
				return IM_COL32(255 * mix, 255 * (1 - mix), 0, 255);
			}();


			ImGui::Text("GPU:");
			ImGui::PushStyleColor(ImGuiCol_Text, temp_color);
			ImGui::Text("%d C", temp);
			ImGui::PopStyleColor();
			//ImGui::Text("%d W", dev.wattage());

			ImGui::EndMenuBar();
		}
		ImGui::End();
	}
	ImGui::PopStyleVar();

	/*if (ImGui::BeginViewportSideBar("##ToolBar", viewport, ImGuiDir_Up, ImGui::GetFrameHeight(), window_flags)) {
		if (ImGui::BeginMenuBar()) {
			ImGui::EndMenuBar();
		}
		ImGui::End();
	}*/
}

std::pair<thunk_t, thunk_t> make_undoer(rfkt::flame& f, const rfkt::descriptor& desc, rfkt::anima&& new_value, rfkt::anima&& old_value) {
	return {
		[&f, desc, new_value = std::move(new_value)]() mutable { *desc.access(f) = new_value; },
		[&f, desc, old_value = std::move(old_value)]() mutable { *desc.access(f) = old_value; }
	};
}

auto make_animator_button(rfkt::flame& f, const rfkt::function_table& ft, std::move_only_function<void(std::pair<thunk_t, thunk_t>&&)>& exec) {
	return [&ft, &exec, &f](const rfkt::descriptor& desc, float width) {

		auto& ani = *desc.access(f);

		if (ImGui::Button(ani.call_info ? "A" : "-", { width , 0 })) {
			ImGui::OpenPopup("animation_editor");
		}

		if (!ImGui::BeginPopup("animation_editor")) { return; }

		std::string cur_selected = ani.call_info ? ani.call_info->first : "none";
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
			rfkt::anima::call_info_t new_call_info = selection == "none" ? std::nullopt : ft.make_default(*selection);
			rfkt::anima::call_info_t old_call_info = ani.call_info;

			exec(make_undoer(f, desc, { ani.t0, std::move(new_call_info) }, { ani.t0, std::move(old_call_info) }));
		}

		if (ani.call_info) {
			bool changed = false;
			rfkt::anima::call_info_t old_call_info = ani.call_info;
			for (auto& [name, arg] : ani.call_info->second) {
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
					else {
						SPDLOG_ERROR("Unhandled case: {}", typeid(T).name());
					}
				}, arg);
			}

			if (changed) {
				rfkt::anima::call_info_t new_call_info = ani.call_info;

				exec(make_undoer(f, desc, { ani.t0, std::move(new_call_info) }, { ani.t0, std::move(old_call_info) }));
			}
		}
		

		ImGui::EndPopup();
	};
}

struct edit_bounds {
	double min = -DBL_MAX;
	double max = DBL_MAX;

	edit_bounds() = default;
	explicit(false) edit_bounds(double min): min{min} {}
	edit_bounds(double min, double max): min{min}, max{max} {}
};

auto make_flame_drag_edit(rfkt::flame& f, std::move_only_function<void(std::pair<thunk_t, thunk_t>&&)>& exec, const rfkt::function_table& ft) {
	auto min_button_width = ImGui::CalcTextSize("A").x + ImGui::GetStyle().FramePadding.x * 2.0f;
	auto animator_button = make_animator_button(f, ft, exec);
	return[&f, &exec, animator_button = std::move(animator_button), min_button_width](const rfkt::descriptor& desc, std::string_view text, float step, const edit_bounds& eb = {}) mutable {
		rfkt::gui::id_scope drag_scope{ desc.hash().str16().c_str() };

		auto* ptr = desc.access(f);
		animator_button(desc, min_button_width);
		ImGui::SameLine();
		if (auto drag_start = rfkt::gui::drag_double(text, ptr->t0, step, eb.min, eb.max); drag_start) {
			rfkt::anima new_value = { ptr->t0, ptr->call_info };
			rfkt::anima old_value = { drag_start.value(), ptr->call_info };
			exec(make_undoer(f, desc, std::move(new_value), std::move(old_value)));
		}
	};
}

bool flame_editor(rfkt::flamedb& fdb, rfkt::flame& f, std::move_only_function<void(std::pair<thunk_t, thunk_t>&&)>& exec, rfkt::function_table& ft) {
	bool modified = false;

	namespace fdesc = rfkt::descriptors;

	rfkt::gui::id_scope flame_scope{ &f };

	auto flame_drag_edit = make_flame_drag_edit(f, exec, ft);

	ImGui::Text("Display");
	flame_drag_edit(fdesc::flame{ &rfkt::flame::rotate}, "Rotate", 0.01);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::scale }, "Scale", 0.1, 0.0001);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::center_x }, "Center X", 0.1);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::center_y }, "Center Y", 0.1);
	ImGui::Separator();

	ImGui::Text("Color");
	flame_drag_edit(fdesc::flame{ &rfkt::flame::gamma }, "Gamma", 0.01, 0.1);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::brightness }, "Brightness", 0.01, 0.1);
	flame_drag_edit(fdesc::flame{ &rfkt::flame::vibrancy }, "Vibrancy", 0.01);
	ImGui::Separator();

	f.for_each_xform([&](int xid, rfkt::xform& xf) {
		rfkt::gui::id_scope xf_scope{ &xf };

		std::string xf_name = (xid == -1) ? "Final Xform" : fmt::format("Xform {}", xid);

		if (ImGui::CollapsingHeader(xf_name.c_str())) {
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::weight }, "Weight", 0.01, 0);
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::color }, "Color", 0.001, { 0, 1 });
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::color_speed }, "Color Speed", 0.001, { 0, 1 });
			flame_drag_edit(fdesc::xform{ xid, &rfkt::xform::opacity }, "Opacity", 0.001, { 0, 1 });

			if (ImGui::BeginTabBar("Vchain")) {
				for (int i = 0; i < xf.vchain.size(); i++) {
					if (ImGui::BeginTabItem(std::format("{}", i).c_str())) {
						//ImGui::BeginChild(std::format("vchain{}", i).c_str(), ImVec2(0, 200));

						auto& vl = xf.vchain[i];
						{
							rfkt::gui::id_scope vl_scope{ &vl };

							ImGui::Text("Affine");
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::a }, "A", 0.001);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::b }, "B", 0.001);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::c }, "C", 0.001);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::d }, "D", 0.001);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::e }, "E", 0.001);
							flame_drag_edit(fdesc::transform{ xid, i, &rfkt::affine::f }, "F", 0.001);
							flame_drag_edit(fdesc::vlink{xid, i, &rfkt::vlink::mod_rotate}, "Rotate", 0.01);
							flame_drag_edit(fdesc::vlink{xid, i, &rfkt::vlink::mod_scale}, "Scale", 0.01, 0.0001);
							flame_drag_edit(fdesc::vlink{xid, i, &rfkt::vlink::mod_x}, "X", 0.01);
							flame_drag_edit(fdesc::vlink{xid, i, &rfkt::vlink::mod_y}, "Y", 0.01);
							ImGui::Separator();

							static bool just_opened = false;
							static char filter[128] = { 0 };

							if (ImGui::Button("Add variation")) {
								ImGui::OpenPopup("add_variation");
								just_opened = true;
								std::memset(filter, 0, sizeof(filter));
							}

							if (ImGui::BeginPopup("add_variation")) {
								if (just_opened) {
									just_opened = false;
									ImGui::SetKeyboardFocusHere(1);
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
									exec({
										[&f, xid, i, &fdb, name = std::string{ selected_var }]() {
											f.get_xform(xid)->vchain[i].add_variation(fdb.make_vardata(name));
										},
										[&f, xid, i, name = std::string{ selected_var }]() {
											f.get_xform(xid)->vchain[i].remove_variation(name);
										}
									});
								}
							}

							std::string_view removed_var = {};
							for (auto& [vname, vd] : vl) {
								rfkt::gui::id_scope var_scope{ &vd };
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
								exec({
									[&f, xid, i, name = std::string{ removed_var }]() {
										f.get_xform(xid)->vchain[i].remove_variation(name);
									},
									[&f, xid, i, name = std::string{ removed_var }, vdata = vl[removed_var]] {
										f.get_xform(xid)->vchain[i].add_variation({ name, vdata });
									}
								});
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

	return modified;
}

namespace midi {
	
	struct message {
		enum class type_t {
			none,
			note_on,
			note_off,
			control_change,
			pitch_bend,
			program_change,
			aftertouch,
			poly_aftertouch
		} type = type_t::none;

		uint8_t channel = 0;
		std::pair<uint8_t, uint8_t> data;

		static message from_bytes(std::span<uint8_t> msg) noexcept {
			message m{};

			if (msg.size() == 0)
				return m;

			m.channel = msg[0] & 0x0F;

			switch (msg[0] & 0xF0) {
			case 0x80:
				m.type = type_t::note_off;
				m.data = { msg[1], msg[2] };
				break;
			case 0x90:
				m.type = type_t::note_on;
				m.data = { msg[1], msg[2] };
				break;
			case 0xA0:
				m.type = type_t::poly_aftertouch;
				m.data = { msg[1], msg[2] };
				break;
			case 0xB0:
				m.type = type_t::control_change;
				m.data = { msg[1], msg[2] };
				break;
			case 0xC0:
				m.type = type_t::program_change;
				m.data = { msg[1], 0 };
				break;
			case 0xD0:
				m.type = type_t::aftertouch;
				m.data = { msg[1], 0 };
				break;
			case 0xE0:
				m.type = type_t::pitch_bend;
				m.data = { msg[1], msg[2] };
				break;
			}

			return m;
		}
	};

	namespace detail {
		RtMidiIn& info_instance() {
			static RtMidiIn instance{};
			return instance;
		}
	};

	auto device_count() {
		return detail::info_instance().getPortCount();
	}

	std::string device_name(unsigned int port) {
		return detail::info_instance().getPortName(port);
	}

}

class midi_system {
public:

private:
	std::map<unsigned int, RtMidiIn> inputs;
};

bool shortcut_pressed(ImGuiKey mod, ImGuiKey key) {
	return ImGui::IsKeyPressed(key, false) && ImGui::IsKeyDown(mod);
}

void main_thread(rfkt::cuda::context ctx, std::stop_source stop) {

	auto c_exec = command_executor{};
	std::move_only_function<void(std::pair<thunk_t, thunk_t>&&)> exec = [&c_exec](std::pair<thunk_t, thunk_t>&& funcs) { return c_exec.execute(std::move(funcs)); };

	namespace ccpp = concurrencpp;

	ctx.make_current();

	rfkt::flamedb fdb;
	rfkt::initialize(fdb, "config");

	auto runtime_path = rfkt::cuda::check_and_download_cudart();
	if (!runtime_path) {
		SPDLOG_CRITICAL("Could not find CUDA runtime");
		return;
	}
	auto kernel_cache = std::make_shared<ezrtc::sqlite_cache>("kernel.sqlite3");
	auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::cache_adaptors::guarded>(kernel_cache));
	ezrtc::compiler kc{ zlib };
	kc.find_system_cuda(*runtime_path);

	rfkt::function_table ft;
	ft.add_or_update("increase", {
		{{"per_loop", {rfkt::func_info::arg_t::decimal, 360.0}}},
		"return iv + t * per_loop"
	});

	ft.add_or_update("sine", {
			{
				{"frequency", {rfkt::func_info::arg_t::decimal, 1.0}},
				{"amplitude", {rfkt::func_info::arg_t::decimal, 1.0}},
				{"phase", {rfkt::func_info::arg_t::decimal, 0.0}},
				{"sharpness", {rfkt::func_info::arg_t::decimal, 0.0}},
				{"absolute", {rfkt::func_info::arg_t::boolean, false}}
			},
			"local v = math.sin(t * frequency * math.pi * 2.0 + math.rad(phase))\n"
			"if sharpness > 0 then v = math.copysign(1.0, v) * (math.abs(v) ^ sharpness) end\n"
			"if absolute then v = math.abs(v) end\n"
			"return iv + v * amplitude\n"
		});

	rfkt::flame_compiler fc(kc);
	auto tm = rfkt::tonemapper{ kc };
	auto dn = rfkt::denoiser{ {2560, 1440}, false };
	auto up_dn = rfkt::denoiser{ {2560, 1440}, true };
	auto conv = rfkt::converter{ kc };

	preview_panel::renderer_t renderer = [&tm, &dn, &up_dn, &conv](
		rfkt::cuda_stream& stream, const rfkt::flame_kernel& kernel,
		rfkt::flame_kernel::saved_state& state,
		rfkt::flame_kernel::bailout_args bo, double3 gbv, bool upscale) {

			const auto total_bins = state.bin_dims.x * state.bin_dims.y;

			const auto output_dims = (upscale) ? uint2{ state.bin_dims.x * 2, state.bin_dims.y * 2 } : state.bin_dims;
			const auto output_bins = output_dims.x * output_dims.y;

			auto tonemapped = rfkt::cuda_buffer<half3>{ total_bins, stream };
			auto denoised = rfkt::cuda_buffer<half3>{ output_bins, stream };
			auto out_buf = rfkt::cuda_buffer<uchar4>{ output_bins, stream };

			auto bin_info = kernel.bin(stream, state, bo).get();
			state.quality += bin_info.quality;

			tm.run(state.bins, tonemapped, state.bin_dims, state.quality, gbv.x, gbv.y, gbv.z, stream);
			auto& denoiser = upscale ? up_dn : dn;
			denoiser.denoise(output_dims, tonemapped, denoised, stream);
			tonemapped.free_async(stream);
			conv.to_24bit(denoised, out_buf, output_dims, false, stream);
			denoised.free_async(stream);
			stream.sync();

			return out_buf;
	};

	auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string("assets/flames/electricsheep.247.47670.flam3"));

	rfkt::gl::make_current();
	rfkt::gl::set_target_fps(144);
	bool should_exit = false;

	ccpp::runtime runtime;

	auto render_worker = runtime.make_worker_thread_executor();
	auto render_stream = rfkt::cuda_stream{};
	render_worker->post([ctx]() {ctx.make_current(); });

	preview_panel::executor_t executor = [render_worker](std::move_only_function<void(void)>&& func) {
		render_worker->post(std::move(func));
	};

	auto thunk_executor = runtime.make_manual_executor();
	thunk_executor->post([&ctx]() {ctx.make_current(); });

	auto prev_panel = preview_panel{render_stream, fc, executor, renderer, exec};
	auto render_modal = rfkt::gui::render_modal{ kc, fc, fdb, ft };


	while (!should_exit) {
		rfkt::gl::begin_frame();
		if (rfkt::gl::close_requested()) should_exit = true;

		while (!thunk_executor->empty()) {
			thunk_executor->loop_once();
		}

		if (shortcut_pressed(ImGuiKey_ModCtrl, ImGuiKey_Z)) c_exec.undo();
		if (shortcut_pressed(ImGuiKey_ModCtrl, ImGuiKey_Y)) c_exec.redo();

		ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(32 / 255.0f, 32 / 255.0f, 32 / 255.0f, 1.0f));

		bool open_render_modal = false;

		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				ImGui::MenuItem("New");

				if (ImGui::MenuItem("Open...")) {
					runtime.background_executor()->enqueue([ctx, &c_exec, &fdb, &thunk_executor, &cur_flame = flame]() mutable {
						ctx.make_current_if_not();

						std::string filename = rfkt::gl::show_open_dialog(rfkt::fs::working_directory() / "assets" / "flames", "Flames\0 * .flam3\0");

						if (filename.empty()) return;

						auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string(filename));
						if (!flame) {
							SPDLOG_ERROR("could not open flame: {}", filename);
							return;
						}

						SPDLOG_INFO("flame data:\n{}", flame->serialize().dump(2));

						thunk_executor->enqueue([&c_exec, flame = std::move(flame), &cur_flame]() mutable {
							c_exec.clear();
							cur_flame = std::move(flame);
						});
					});
				}

				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Edit")) {
				if (ImGui::MenuItem("Undo", "CTRL+Z", false, c_exec.can_undo())) {
					c_exec.undo();
				}
				if (ImGui::MenuItem("Redo", "CTRL+Y", false, c_exec.can_redo())) {
					c_exec.redo();
				}
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Render")) {
				if (ImGui::MenuItem("Video")) {
					open_render_modal = true;
				}
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Style")) {
				auto styles = rfkt::gui::get_styles();

				for (const auto& style : styles) {
					if (ImGui::MenuItem(style.name.c_str())) {
						rfkt::gui::set_style(style.id);
					}
				}
				ImGui::EndMenu();
			}

			if (ImGui::BeginMenu("Debug")) {
				if (ImGui::MenuItem("Copy flame source")) {
					rfkt::gl::set_clipboard(fc.make_source(fdb, flame.value()));
				}
				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}
		ImGui::PopStyleVar();
		ImGui::PopStyleColor();

		if (ImGui::Begin("Flame Options")) {
			flame_editor(fdb, flame.value(), exec, ft);
		}
		ImGui::End();

		ImGui::ShowDemoWindow();
		draw_status_bar();

		prev_panel.show(fdb, flame.value(), ft);
		if(open_render_modal) render_modal.show(flame.value());
		render_modal.frame_logic();

		rfkt::gl::end_frame(true);
	}

	stop.request_stop();
	glfwPostEmptyEvent();
}

int main(int argc, char**) {



	auto ctx = rfkt::cuda::init();
	rfkt::denoiser::init(ctx);
	rfkt::gpuinfo::init();

	if (!rfkt::gl::init(1920, 1080)) {
		return 1;
	}
	rfkt::gui::set_style(3);

	std::stop_source stopper;
	auto stoke = stopper.get_token();

	auto render_thread = std::jthread(main_thread, ctx, std::move(stopper));

	rfkt::gl::event_loop(stoke);

	

	return 0;
}