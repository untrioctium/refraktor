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

#include "gl.h"
#include "gui.h"

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
		SPDLOG_INFO("undoing");
		--current_command;
		current_command->undo();
	}

	void redo() {
		if (!can_redo())
			return;
		SPDLOG_INFO("redoing");
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

std::pair<thunk_t, thunk_t> make_flame_undoer(rfkt::flame& f, double rfkt::flame::* ptr, double new_value, double old_value) {
	return {
		[&f, new_value, ptr]() mutable { f.*ptr = new_value; },
		[&f, old_value, ptr]() mutable { f.*ptr = old_value; }
	};
}

std::pair<thunk_t, thunk_t> make_xform_undoer(rfkt::flame& f, int xid, double rfkt::xform::* ptr, double new_value, double old_value) {
	return {
		[&f, xid, new_value, ptr]() mutable { f.get_xform(xid)->*ptr = new_value; },
		[&f, xid, old_value, ptr]() mutable { f.get_xform(xid)->*ptr = old_value; }
	};
}

std::pair<thunk_t, thunk_t> make_vlink_undoer(rfkt::flame& f, int xid, int vid, double rfkt::vlink::* ptr, double new_value, double old_value) {
	return {
		[&f, xid, vid, new_value, ptr]() mutable { f.get_xform(xid)->vchain[vid].*ptr = new_value; },
		[&f, xid, vid, old_value, ptr]() mutable { f.get_xform(xid)->vchain[vid].*ptr = old_value; }
	};
}

std::pair<thunk_t, thunk_t> make_transform_undoer(rfkt::flame& f, int xid, int vid, double rfkt::affine::* ptr, double new_value, double old_value) {
	return {
		[&f, xid, vid, new_value, ptr]() mutable { f.get_xform(xid)->vchain[vid].transform.*ptr = new_value; },
		[&f, xid, vid, old_value, ptr]() mutable { f.get_xform(xid)->vchain[vid].transform.*ptr = old_value; }
	};
}

std::pair<thunk_t, thunk_t> make_vardata_undoer(rfkt::flame& f, int xid, int vid, std::string_view vname, double rfkt::vardata::* ptr, double new_value, double old_value) {
	return {
		[&f, xid, vid, vname = std::string{vname}, new_value, ptr]() mutable { f.get_xform(xid)->vchain[vid][vname].*ptr = new_value; },
		[&f, xid, vid, vname = std::string{vname}, old_value, ptr]() mutable { f.get_xform(xid)->vchain[vid][vname].*ptr = old_value; }
	};
}

std::pair<thunk_t, thunk_t> make_parameter_undoer(rfkt::flame& f, int xid, int vid, std::string_view vname, std::string_view pname, double new_value, double old_value) {
	return {
		[&f, xid, vid, vname = std::string{vname}, pname = std::string{pname}, new_value]() mutable { f.get_xform(xid)->vchain[vid][vname][pname] = new_value; },
		[&f, xid, vid, vname = std::string{vname}, pname = std::string{pname}, old_value]() mutable { f.get_xform(xid)->vchain[vid][vname][pname] = old_value; }
	};
}

bool flame_editor(rfkt::flamedb& fdb, rfkt::flame& f, std::move_only_function<void(std::pair<thunk_t, thunk_t>&&)>& exec) {
	bool modified = false;

	rfkt::gui::id_scope flame_scope{ &f };

	ImGui::Text("Display");
	if(auto drag_start = rfkt::gui::drag_double("Rotate", f.rotate, 0.01, -360, 360); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::rotate, f.rotate, drag_start.value()));

	if(auto drag_start = rfkt::gui::drag_double("Scale", f.scale, 0.1, 0, 10'000); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::scale, f.scale, drag_start.value()));

	if(auto drag_start = rfkt::gui::drag_double("Center X", f.center_x, 0.1, -10, 10); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::center_x, f.center_x, drag_start.value()));

	if(auto drag_start = rfkt::gui::drag_double("Center Y", f.center_y, 0.1, -10, 10); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::center_y, f.center_y, drag_start.value()));

	ImGui::Separator();

	ImGui::Text("Color");
	if(auto drag_start = rfkt::gui::drag_double("Gamma", f.gamma, 0.01, 0, 5); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::gamma, f.gamma, drag_start.value()));

	if(auto drag_start = rfkt::gui::drag_double("Brightness", f.brightness, 0.01, 0, 100); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::brightness, f.brightness, drag_start.value()));

	if(auto drag_start = rfkt::gui::drag_double("Vibrancy", f.vibrancy, 0.1, 0, 100); drag_start)
		exec(make_flame_undoer(f, &rfkt::flame::vibrancy, f.vibrancy, drag_start.value()));

	ImGui::Separator();

	f.for_each_xform([&](int xid, rfkt::xform& xf) {
		rfkt::gui::id_scope xf_scope{ &xf };

		std::string xf_name = (xid == -1) ? "Final Xform" : fmt::format("Xform {}", xid);

		if (ImGui::CollapsingHeader(xf_name.c_str())) {
			if(auto drag_start = rfkt::gui::drag_double("Weight", xf.weight, 0.01, 0, 50); drag_start)
				exec(make_xform_undoer(f, xid, &rfkt::xform::weight, xf.weight, drag_start.value()));

			if(auto drag_start = rfkt::gui::drag_double("Color", xf.color, 0.001, 0, 1); drag_start)
				exec(make_xform_undoer(f, xid, &rfkt::xform::color, xf.color, drag_start.value()));

			if(auto drag_start = rfkt::gui::drag_double("Color Speed", xf.color_speed, 0.001, 0, 1))
				exec(make_xform_undoer(f, xid, &rfkt::xform::color_speed, xf.color_speed, drag_start.value()));

			if(auto drag_start = rfkt::gui::drag_double("Opacity", xf.opacity, 0.001, 0, 1); drag_start)
				exec(make_xform_undoer(f, xid, &rfkt::xform::opacity, xf.opacity, drag_start.value()));

			if (ImGui::BeginTabBar("Vchain")) {
				for (int i = 0; i < xf.vchain.size(); i++) {
					if (ImGui::BeginTabItem(std::format("{}", i).c_str())) {
						//ImGui::BeginChild(std::format("vchain{}", i).c_str(), ImVec2(0, 200));

						auto& vl = xf.vchain[i];
						{
							rfkt::gui::id_scope vl_scope{ &vl };

							if(auto drag_start = rfkt::gui::drag_double("Affine A", vl.transform.a, 0.001, -5, 5); drag_start)
								exec(make_transform_undoer(f, xid, i, &rfkt::affine::a, vl.transform.a, drag_start.value()));

							if(auto drag_start = rfkt::gui::drag_double("Affine B", vl.transform.b, 0.001, -5, 5); drag_start)
								exec(make_transform_undoer(f, xid, i, &rfkt::affine::b, vl.transform.b, drag_start.value()));

							if(auto drag_start = rfkt::gui::drag_double("Affine C", vl.transform.c, 0.001, -5, 5); drag_start)
								exec(make_transform_undoer(f, xid, i, &rfkt::affine::c, vl.transform.c, drag_start.value()));

							if(auto drag_start = rfkt::gui::drag_double("Affine D", vl.transform.d, 0.001, -5, 5); drag_start)
								exec(make_transform_undoer(f, xid, i, &rfkt::affine::d, vl.transform.d, drag_start.value()));

							if(auto drag_start = rfkt::gui::drag_double("Affine E", vl.transform.e, 0.001, -5, 5); drag_start)
								exec(make_transform_undoer(f, xid, i, &rfkt::affine::e, vl.transform.e, drag_start.value()));

							if(auto drag_start = rfkt::gui::drag_double("Affine F", vl.transform.f, 0.001, -5, 5); drag_start)
								exec(make_transform_undoer(f, xid, i, &rfkt::affine::f, vl.transform.f, drag_start.value()));

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
								if (auto drag_start = rfkt::gui::drag_double(vname.c_str(), vd.weight, 0.001, -50, 50); drag_start)
									exec(make_vardata_undoer(f, xid, i, vname, &rfkt::vardata::weight, vd.weight, drag_start.value()));
							
								if (vl.size_variations() > 1) {
									ImGui::SameLine(ImGui::GetContentRegionAvail().x - 10);
									if (ImGui::Button("X")) removed_var = vname;
								}

								for (auto& [pname, val] : vd) {
									if (auto drag_start = rfkt::gui::drag_double(std::format("{}_{}", vname, pname).c_str(), val, 0.001, -50, 50); drag_start)
										exec(make_parameter_undoer(f, xid, i, vname, pname, val, drag_start.value()));
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

	auto kernel_cache = std::make_shared<ezrtc::sqlite_cache>("kernel.sqlite3");
	auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::cache_adaptors::guarded>(kernel_cache));

	ezrtc::compiler kc{ zlib };
	//kc.find_system_cuda();

	rfkt::flame_compiler fc(kc);
	auto tm = rfkt::tonemapper{ kc };
	auto dn = rfkt::denoiser{ {2560, 1440}, false };
	auto conv = rfkt::converter{ kc };

	preview_panel::renderer_t renderer = [&tm, &dn, &conv](
		rfkt::cuda_stream& stream, const rfkt::flame_kernel& kernel,
		rfkt::flame_kernel::saved_state& state,
		rfkt::flame_kernel::bailout_args bo, double3 gbv) {

			const auto total_bins = state.bin_dims.x * state.bin_dims.y;

			auto out_buf = rfkt::cuda_buffer<uchar4>{ total_bins, stream };
			auto tonemapped = rfkt::cuda_buffer<half3>{ total_bins, stream };
			auto denoised = rfkt::cuda_buffer<half3>{ total_bins, stream };

			auto bin_info = kernel.bin(stream, state, bo).get();
			state.quality += bin_info.quality;

			tm.run(state.bins, tonemapped, state.bin_dims, state.quality, gbv.x, gbv.y, gbv.z, stream);
			dn.denoise(state.bin_dims, tonemapped, denoised, stream);
			tonemapped.free_async(stream);
			conv.to_24bit(denoised, out_buf, state.bin_dims, false, stream);
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
		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				ImGui::MenuItem("New");

				if (ImGui::MenuItem("Open...")) {
					runtime.background_executor()->enqueue([ctx, &c_exec, &fdb, &thunk_executor, &cur_flame = flame]() mutable {
						ctx.make_current_if_not();

						std::string filename = rfkt::gl::show_open_dialog("Flames\0*.flam3\0");

						if (filename.empty()) return;

						auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string(filename));
						if (!flame) {
							SPDLOG_ERROR("could not open flame: {}", filename);
							return;
						}

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

			/*if (ImGui::BeginMenu("Style")) {
				auto styles = rfkt::gui::get_styles();

				for (const auto& style : styles) {
					if (ImGui::MenuItem(style.name.c_str())) {
						rfkt::gui::set_style(style.id);
					}
				}
				ImGui::EndMenu();
			}*/

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
			flame_editor(fdb, flame.value(), exec);
		}
		ImGui::End();

		//ImGui::ShowDemoWindow();
		draw_status_bar();

		prev_panel.show(fdb, flame.value());

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