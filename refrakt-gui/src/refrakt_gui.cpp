#include <cuda.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <concurrencpp/concurrencpp.h>

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
			ImGui::Text("%d W", dev.wattage());

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

bool flame_editor(rfkt::flamedb& fdb, rfkt::flame& f) {
	bool modified = false;

	rfkt::gui::id_scope flame_scope{ &f };

	ImGui::Text("Display");
	modified |= rfkt::gui::drag_double("Rotate", f.rotate, 0.01, -360, 360);
	modified |= rfkt::gui::drag_double("Scale", f.scale, 0.1, 0, 10'000);
	modified |= rfkt::gui::drag_double("Center X", f.center_x, 0.1, -10, 10);
	modified |= rfkt::gui::drag_double("Center Y", f.center_y, 0.1, -10, 10);
	ImGui::Separator();

	ImGui::Text("Color");
	modified |= rfkt::gui::drag_double("Gamma", f.gamma, 0.01, 0, 5);
	modified |= rfkt::gui::drag_double("Brightness", f.brightness, 0.01, 0, 100);
	modified |= rfkt::gui::drag_double("Vibrancy", f.vibrancy, 0.1, 0, 100);
	ImGui::Separator();

	f.for_each_xform([&](int xid, rfkt::xform& xf) {
		rfkt::gui::id_scope xf_scope{ &xf };

		std::string xf_name = (xid == -1) ? "Final Xform" : fmt::format("Xform {}", xid);

		if (ImGui::CollapsingHeader(xf_name.c_str())) {
			modified |= rfkt::gui::drag_double("Weight", xf.weight, 0.01, 0, 50);
			modified |= rfkt::gui::drag_double("Color", xf.color, 0.001, 0, 1);
			modified |= rfkt::gui::drag_double("Color Speed", xf.color_speed, 0.001, 0, 1);
			modified |= rfkt::gui::drag_double("Opacity", xf.opacity, 0.001, 0, 1);

			if (ImGui::BeginTabBar("Vchain")) {
				for (int i = 0; i < xf.vchain.size(); i++) {
					if (ImGui::BeginTabItem(std::format("{}", i).c_str())) {
						ImGui::BeginChild(std::format("vchain{}", i).c_str(), ImVec2(0, 200));

						auto& vl = xf.vchain[i];
						{
							rfkt::gui::id_scope vl_scope{ &vl };

							modified |= rfkt::gui::drag_double("Affine A", vl.transform.a, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine B", vl.transform.b, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine C", vl.transform.c, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine D", vl.transform.d, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine E", vl.transform.e, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine F", vl.transform.f, 0.001, -5, 5);
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
									ImGui::SetKeyboardFocusHere(-1);
								}

								ImGui::InputText("##filter", filter, sizeof(filter));
								std::string_view selected_var{};

								if (ImGui::BeginListBox("##vlist", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing()))) {
									for (const auto& var : fdb.variations()) {
										if (!vl.has_variation(var.name) && (strlen(filter) < 2 || var.name.find(filter) != std::string::npos)) {
											if (ImGui::Selectable(var.name.c_str())) {
												selected_var = var.name;
											}
										}
									}
									ImGui::EndListBox();
								}
								ImGui::EndPopup();
							}

							for (auto& [vname, vd] : vl) {
								ImGui::Separator();
								modified |= rfkt::gui::drag_double(vname.c_str(), vd.weight, 0.001, -50, 50);

								for (auto& [pname, val] : vd) {
									modified |= rfkt::gui::drag_double(std::format("{}_{}", vname, pname).c_str(), val, 0.001, -50, 50);
								}
							}
						}
						ImGui::EndChild();
						ImGui::EndTabItem();
					}
				}
				ImGui::EndTabBar();
			}
		}
	});

	return modified;
}

void main_thread(rfkt::cuda::context ctx, std::stop_source stop) {

	namespace ccpp = concurrencpp;

	ctx.make_current();

	rfkt::flamedb fdb;
	rfkt::initialize(fdb, "config");

	auto kernel_cache = std::make_shared<ezrtc::sqlite_cache>("kernel.sqlite3");
	auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::cache_adaptors::guarded>(kernel_cache));

	ezrtc::compiler kc{ zlib };
	kc.find_system_cuda();

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

	auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string("assets/flames_test/electricsheep.247.47670.flam3"));

	rfkt::gl::make_current();
	rfkt::gl::set_target_fps(120);
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

	auto prev_panel = preview_panel{render_stream, fc, executor, renderer};

	while (!should_exit) {
		rfkt::gl::begin_frame();
		if (rfkt::gl::close_requested()) should_exit = true;

		while (!thunk_executor->empty()) {
			thunk_executor->loop_once();
		}

		ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(32 / 255.0f, 32 / 255.0f, 32 / 255.0f, 1.0f));
		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				ImGui::MenuItem("New");

				if (ImGui::MenuItem("Open...")) {
					runtime.background_executor()->enqueue([ctx, &fdb, &thunk_executor, &cur_flame = flame]() mutable {
						ctx.make_current_if_not();

						std::string filename = rfkt::gl::show_open_dialog("Flames\0*.flam3\0");

						if (filename.empty()) return;

						auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string(filename));
						if (!flame) {
							SPDLOG_ERROR("could not open flame: {}", filename);
						}

						thunk_executor->enqueue([flame = std::move(flame), &cur_flame]() mutable {
							cur_flame = std::move(flame);
						});
					});
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
			flame_editor(fdb, flame.value());
		}
		ImGui::End();

		ImGui::ShowDemoWindow();
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