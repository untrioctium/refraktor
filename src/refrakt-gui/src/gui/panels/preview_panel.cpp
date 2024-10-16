#include <numbers>

#include <imftw/imftw.h>
#include <imftw/gui.h>

#include "gui/panels/preview_panel.h"

#include <IconsMaterialDesign.h>

#include <implot.h>

preview_panel::~preview_panel()
{
	if (rendering_texture) {
		auto tex = rendering_texture->get();
		//cuda_map = std::nullopt;
	}
}

bool preview_panel::show(const rfkt::flamedb& fdb, rfkt::flame& flame, rfkt::function_table& ft)
{

	if (render_is_ready()) {
		displayed_texture = rendering_texture->get();
		//cuda_map = std::nullopt;
		rendering_texture = std::nullopt;
	}

	auto preview_size = gui_logic(flame, ft);

	if (upscale && denoise) {
		if (preview_size.x % 2 == 1) preview_size.x -= 1;
		if (preview_size.y % 2 == 1) preview_size.y -= 1;
	}

	const auto given_struct_hash = flame.hash();
	const auto given_value_hash = flame.value_hash();
	const auto given_fdb_hash = fdb.hash();

	bool needs_kernel = (given_struct_hash != flame_structure_hash) || given_fdb_hash != flamedb_hash;
	bool needs_clear =
		needs_kernel
		|| playing
		|| render_options_changed
		|| (given_value_hash != flame_value_hash)
		|| !displayed_texture
		|| (displayed_texture && (preview_size.x != render_dims.x || preview_size.y != render_dims.y) && preview_size.x && preview_size.y);

	if (rendering_texture && needs_clear && current_state) {
		SPDLOG_INFO("Aborting binning kernel");
		current_state->abort_binning();
	}

	if (!rendering_texture) {

		bool needs_render = needs_clear || (current_state && current_state->quality < target_quality);

		if (needs_render) {
			auto state_size = (upscale && denoise) ? uint2{ preview_size.x / 2, preview_size.y / 2 } : preview_size;
			auto out_tex = texture_t::element_type::create(preview_size.x, preview_size.y, rfkt::gl::sampling_mode::nearest);
			auto invoker = ft.make_invoker();

			std::vector<double> samples{};
			if (needs_clear) {
				auto packer = [&samples](double v) { samples.push_back(v); };

				const auto loops_per_frame = 1 / 150.0;

				if(animate)
					flame.pack_samples(packer, invoker, current_time - loops_per_frame, loops_per_frame, 4, state_size.x, state_size.y);
				else
					flame.pack_samples(packer, invoker, current_time, 0, 4, state_size.x, state_size.y);
			}

			std::optional<rfkt::flame> flame_copy = std::nullopt;
			if (needs_kernel) {
				flame_copy = flame;
			}

			auto promise = std::promise<texture_t>();
			auto future = promise.get_future();

			auto cuda_map = out_tex->map_to_cuda();

			submitter([
				cuda_map = std::move(cuda_map),
				promise = std::move(promise),
				flame_copy = std::move(flame_copy),
				&renderer = this->renderer,
				out_tex = std::move(out_tex),
				needs_kernel, needs_clear,
				samples = std::move(samples),
				&compiler = this->compiler, kernel = this->kernel, &fdb,
				state = this->current_state,
				gbv = double3{ flame.gamma.sample(current_time, invoker), flame.brightness.sample(current_time, invoker), flame.vibrancy.sample(current_time, invoker)},
				&stream = this->stream, preview_size,
				target_quality = this->target_quality,
				upscale = this->upscale,
				denoise = this->denoise,
				temporal_multiplier = this->temporal_multiplier,
				&densities = this->densities]() mutable noexcept {

					if (needs_kernel) {
						auto result = compiler.get_flame_kernel(fdb, rfkt::precision::f32, *flame_copy);

						SPDLOG_INFO("Source:\n{}", result.source);

						if(!result.kernel.has_value()) {
							SPDLOG_ERROR("Failed to compile kernel: {}", result.log);
							exit(1);
						}

						*kernel = std::move(result.kernel.value());
					}

					auto millis = 1000u;

					if (needs_clear) {
						auto state_size = (upscale && denoise) ? uint2{ preview_size.x / 2, preview_size.y / 2 } : preview_size;
						*state = kernel->warmup(stream, samples, state_size, 0xdeadbeef, 1000, temporal_multiplier);
						SPDLOG_INFO("Created state of size: {} MB", state->shared.size_bytes() / (1024.0 * 1024.0));
					}
					else {
						//millis = 100u;
					}

					auto result = renderer(stream, *kernel, *state, { .millis = millis, .quality = target_quality - state->quality }, gbv, upscale && denoise, denoise);
					cuda_map.copy_from(result, stream);
					result.free_async(stream);

					/*auto histo = state->density_histogram.to_host();

					auto total = static_cast<double>(std::accumulate(histo.begin(), histo.end(), 0));
					double cdf = 0.0;

					for(auto i = 0; i < histo.size(); i++) {
						auto pct = static_cast<double>(histo[i]) / total * 100;
						cdf += pct;
						SPDLOG_INFO("Density histogram[{}]: {:.3} {:.3} cumulative", i, pct, cdf);
					}*/
					
					stream.host_func(
						ImFtw::MakeDeferer(
							[promise = std::move(promise), out_tex = std::move(out_tex), cuda_map = std::move(cuda_map), state, &densities]() mutable {
								densities = state->density_histogram.to_host();
								promise.set_value(std::move(out_tex));
							}
						)
					);
				});

			rendering_texture = std::move(future);
			flame_structure_hash = given_struct_hash;
			flame_value_hash = given_value_hash;
			flamedb_hash = given_fdb_hash;
			render_dims = preview_size;
			render_options_changed = false;
		}

	}



	return false;
}

uint2 preview_panel::gui_logic(rfkt::flame& flame, rfkt::function_table& ft) {

	//rfkt::gui::scope::style padding_scope = { ImGuiStyleVar_WindowPadding, ImVec2(0, 0) };

	uint2 preview_size = { 0, 0 };
	bool preview_hovered = false;
	bool render_changed = false;

	auto invoker = ft.make_invoker();

	IMFTW_WINDOW("\xee\xa9\x83" " Render##preview_panel", ImGuiWindowFlags_MenuBar) {
		IMFTW_MENU_BAR() {
			IMFTW_MENU("Render options") {
				if (ImGui::MenuItem("Upscale 2x", nullptr, &this->upscale)) { 
					render_options_changed = true;
				}
				ImFtw::Tooltip("Enables upscaling; improves performance but reduces quality.", false);
			}

			IMFTW_MENU("Quality") {
				const static std::vector<std::pair<std::string, double>> qualities{
					{"Terrible", 2},
					{"Very Low", 8},
					{"Low", 32},
					{"Medium", 128},
					{"High", 512},
					{"Ultra", 2048},
					{"Unlimited", std::pow(2.0, 31.0)}
				};

				for (const auto& [name, quality]: qualities) {
					if (ImGui::MenuItem(name.c_str(), nullptr, target_quality == quality) && quality != target_quality) {
						target_quality = quality;
						render_options_changed = true;
					}
				}
			}

			IMFTW_MENU("Aspect Ratio") {

				const static std::vector<std::pair<std::string_view, double>> ratios{
					{"Fit to Window", 0.0},
					{"1:1", 1},
					{"5:4", 5.0 / 4.0},
					{"4:3", 4.0 / 3.0},
					{"3:2", 3.0 / 2.0},
					{"16:10", 16.0 / 10.0},
					{"16:9", 16.0 / 9.0},
					{"21:9", 21.0 / 9.0},
					{"32:9", 32.0 / 9.0}
				};

				for (const auto& [name, ratio] : ratios) {
					if (ImGui::MenuItem(name.data(), nullptr, aspect_ratio == ratio) && aspect_ratio != ratio) {
						aspect_ratio = ratio;
						render_options_changed = true;
					}
				}

			}

			IMFTW_MENU("Multiplier") {
				for (int i = 1; i < 16; i++) {
					if (ImGui::MenuItem(std::to_string(i).c_str(), nullptr, temporal_multiplier == i) && temporal_multiplier != i) {
						temporal_multiplier = i;
						render_options_changed = true;
					}
				}
			}

			//ImGui::InputFloat("Gamma", &gamma);
			//ImGui::InputFloat("Exposure", &exposure);

			render_options_changed |= ImGui::Checkbox("Animate", &animate);
			render_options_changed |= ImGui::Checkbox("Denoise", &denoise);
		}

		static double tmin = 0.0;
		static double tmax = 1.0;

		constexpr static auto play_id = ICON_MD_PLAY_ARROW "##toggle_playing";
		constexpr static auto stop_id = ICON_MD_STOP "##toggle_playing";

		ImGui::PushStyleColor(ImGuiCol_Text, (playing) ? ImVec4(0.8f, 0.0f, 0.0f, 1.0f) : ImVec4(0.0f, 0.8f, 0.0f, 1.0f));
		if (ImGui::Button(playing ? stop_id: play_id)) playing = !playing;
		ImGui::PopStyleColor();

		ImGui::SameLine();
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderScalar("##Time", ImGuiDataType_Double, &current_time, &tmin, &tmax)) {
			render_options_changed = true;
		}
		ImGui::PopItemWidth();

		auto avail = ImGui::GetContentRegionAvail();

		if (avail.x < 0) avail.x = 10;
		if (avail.y < 0) avail.y = 10;

		preview_size = { static_cast<unsigned int>(avail.x), static_cast<unsigned int>(avail.y) };

		if (aspect_ratio != 0.0) {
			auto old_preview = preview_size;

			preview_size.x = static_cast<uint32_t>(preview_size.y * aspect_ratio);
			if (preview_size.x > old_preview.x) {
				preview_size.x = old_preview.x;
				preview_size.y = static_cast<uint32_t>(preview_size.x / aspect_ratio);
			}
		}

		if (displayed_texture.has_value()) {
			auto& tex = displayed_texture.value();

			auto draw_size = preview_size;
			if (draw_size.x % 2 == 1) draw_size.x -= 1;
			if (draw_size.y % 2 == 1) draw_size.y -= 1;

			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail.x - draw_size.x) / 2.0);
			ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (avail.y - draw_size.y) / 2.0);

			/*ImGui::GetWindowDrawList()->AddCallback(
				[](const ImDrawList* parent_list, const ImDrawCmd* cmd) {
					auto panel = reinterpret_cast<preview_panel*>(cmd->UserCallbackData);
					glUniform1f(1, panel->gamma);
					glUniform1f(2, panel->exposure);
				},
				this
			);*/

			ImGui::Image((void*)(intptr_t)tex->id(), ImVec2(draw_size.x, draw_size.y));

			/*ImGui::GetWindowDrawList()->AddCallback(
				[](const ImDrawList* parent_list, const ImDrawCmd* cmd) {
					glUniform1f(1, 2.2f);
					glUniform1f(2, 2.0f);
				},
				nullptr
			);*/

			preview_hovered = ImGui::IsItemHovered();
		}
	}

	IMFTW_WINDOW("Stats") {
		if (densities.size() > 0) {

			auto densities_pcts = std::vector<double>(densities.size());
			auto densities_cdf = std::vector<double>(densities.size());

			auto total = static_cast<double>(std::accumulate(densities.begin() + 1, densities.end(), 0));
			
			for(auto i = 1; i < densities.size(); i++) {
				densities_pcts[i] = static_cast<double>(densities[i]) / total * 100;
				densities_cdf[i] = densities_pcts[i];
				if(i > 1) densities_cdf[i] += densities_cdf[i - 1];
			}

			if(ImPlot::BeginPlot("Density Histogram", "Density", "Percentage")) {
				ImPlot::PlotBars("Density", densities_pcts.data() + 1, densities.size() - 1, 1.0, 0.0, 0);
				ImPlot::PlotLine("CDF", densities_cdf.data() + 1, densities.size() - 1);
				
				const static double guides[] = { 50.0, 75.0, 99.0 };
				ImPlot::PlotInfLines("Guides", guides, 3, ImPlotInfLinesFlags_Horizontal);

				ImPlot::EndPlot();
			}
		}
	}

	if (!dragging && preview_hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
		dragging = true;
		drag_start = { flame.center_x.t0, flame.center_y.t0 };
		last_delta = ImGui::GetMouseDragDelta();
	}
	else if (dragging) {
		if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
			dragging = false;

			cmd_exec.execute(
				[&flame, new_value = double2(flame.center_x.t0, flame.center_y.t0)]() mutable {
					flame.center_x.t0 = new_value.x;
					flame.center_y.t0 = new_value.y;
				},
				[&flame, old_value = drag_start]() mutable {
					flame.center_x.t0 = old_value.x;
					flame.center_y.t0 = old_value.y;
				}
			);
		}
		else {
			auto delta = ImGui::GetMouseDragDelta();
			auto drag_dist = ImVec2{ delta.x - last_delta.x, delta.y - last_delta.y };
			if (drag_dist.x != 0 || drag_dist.y != 0) {
				auto scale = flame.scale.t0;
				auto rot = -flame.rotate.t0 * std::numbers::pi / 180;

				double2 vector = { drag_dist.x / (scale * preview_size.y) , drag_dist.y / (scale * preview_size.y) };
				double2 rotated = { vector.x * cos(rot) - vector.y * sin(rot), vector.x * sin(rot) + vector.y * cos(rot) };

				flame.center_x.t0 -= rotated.x;
				flame.center_y.t0 -= rotated.y;
			}

			last_delta = delta;
		}
	}

	if (preview_hovered) {
		ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
	}

	if (auto scroll = ImGui::GetIO().MouseWheel; scroll != 0 && preview_hovered && !dragging) {
		flame.scale.t0 *= std::pow(1.1, scroll);
	}

	return preview_size;
}