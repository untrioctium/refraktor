#include <numbers>

#include <imftw/gui.h>

#include "gui/panels/preview_panel.h"

preview_panel::~preview_panel()
{
	if (rendering_texture) {
		auto tex = rendering_texture->get();
		cuda_map = std::nullopt;
	}
}

bool preview_panel::show(const rfkt::flamedb& fdb, rfkt::flame& flame, rfkt::function_table& ft)
{

	if (render_is_ready()) {
		displayed_texture = rendering_texture->get();
		cuda_map = std::nullopt;
		rendering_texture = std::nullopt;
	}

	auto preview_size = gui_logic(flame);

	if (!rendering_texture) {

		if (upscale) {
			if (preview_size.x % 2 == 1) preview_size.x += 1;
			if (preview_size.y % 2 == 1) preview_size.y += 1;
		}

		const auto given_struct_hash = flame.hash();
		const auto given_value_hash = get_value_hash(flame);
		const auto given_fdb_hash = fdb.hash();

		bool needs_kernel = (given_struct_hash != flame_structure_hash) || given_fdb_hash != flamedb_hash;
		bool needs_clear =
			needs_kernel
			|| render_options_changed
			|| (given_value_hash != flame_value_hash)
			|| !displayed_texture
			|| (displayed_texture && (preview_size.x != render_dims.x || preview_size.y != render_dims.y) && preview_size.x && preview_size.y);

		bool needs_render = needs_clear || (current_state && current_state->quality < target_quality);

		if (needs_render) {
			auto state_size = upscale ? uint2{ preview_size.x / 2, preview_size.y / 2 } : preview_size;
			auto out_tex = rfkt::gl::texture{ preview_size.x, preview_size.y };
			cuda_map = out_tex.map_to_cuda();

			auto invoker = [&ft]<typename... Args>(Args&&... args) { return ft.call(std::forward<Args>(args)...); };

			std::vector<double> samples{};
			if (needs_clear) {
				auto packer = [&samples](double v) { samples.push_back(v); };

				const auto loops_per_frame = 1 / 150.0;

				flame.pack_sample(packer, invoker, current_time - loops_per_frame, state_size.x, state_size.y);
				flame.pack_sample(packer, invoker, current_time, state_size.x, state_size.y);
				flame.pack_sample(packer, invoker, current_time + loops_per_frame, state_size.x, state_size.y);
				flame.pack_sample(packer, invoker, current_time + 2 * loops_per_frame, state_size.x, state_size.y);
			}

			std::optional<rfkt::flame> flame_copy = std::nullopt;
			if (needs_kernel) {
				flame_copy = flame;
			}

			//std::vector<rfkt::flame> samples = needs_clear ? gen_samples2(flame, current_time, 1.0 / 300) : std::vector<rfkt::flame>{};

			auto promise = std::promise<rfkt::gl::texture>();
			auto future = promise.get_future();

			submitter([promise = std::move(promise),
				flame_copy = std::move(flame_copy),
				&renderer = this->renderer,
				out_tex = std::move(out_tex), &cuda_map = this->cuda_map.value(),
				needs_kernel, needs_clear,
				samples = std::move(samples),
				&compiler = this->compiler, kernel = this->kernel, &fdb,
				state = this->current_state,
				gbv = double3{ flame.gamma.sample(current_time, invoker), flame.brightness.sample(current_time, invoker), flame.vibrancy.sample(current_time, invoker)},
				&stream = this->stream, preview_size,
				target_quality = this->target_quality,
				upscale = this->upscale]() mutable {

					if (needs_kernel) {
						auto result = compiler.get_flame_kernel(fdb, rfkt::precision::f32, *flame_copy);
						*kernel = std::move(result.kernel.value());
					}

					if (needs_clear) {
						auto state_size = upscale ? uint2{ preview_size.x / 2, preview_size.y / 2 } : preview_size;
						*state = kernel->warmup(stream, samples, state_size, 0xdeadbeef, 100);
					}

					auto millis = 1000u / 60;
					if (target_quality > 10'000) {
						millis = 100;
					}

					auto result = renderer(stream, *kernel, *state, { .millis = millis, .quality = target_quality - state->quality }, gbv, upscale);
					cuda_map.copy_from(result, stream);
					result.free_async(stream);
					stream.sync();
					promise.set_value(std::move(out_tex));

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

uint2 preview_panel::gui_logic(rfkt::flame& flame) {

	//rfkt::gui::scope::style padding_scope = { ImGuiStyleVar_WindowPadding, ImVec2(0, 0) };

	uint2 preview_size = { 0, 0 };
	bool preview_hovered = false;
	bool render_changed = false;
	IMFTW_WINDOW("Render", ImGuiWindowFlags_MenuBar) {
		IMFTW_MENU_BAR() {
			IMFTW_MENU("Render options") {
				if (ImGui::MenuItem("Upscale 2x", nullptr, &this->upscale)) { 
					render_options_changed = true;
				}
				imftw::tooltip("Enables upscaling; improves performance but reduces quality.", false);
			}

			IMFTW_MENU("Quality") {
				const static std::vector<std::pair<std::string, double>> qualities{
					{"Low", 32},
					{"Medium", 128},
					{"High", 512},
					{"Ultra", 2048},
					{"Unlimited", std::pow(2.0, 31.0)}
				};

				for (const auto& [name, quality]: qualities) {
					if (ImGui::MenuItem(name.c_str())) {
						target_quality = quality;
						render_options_changed = true;
					}
				}
			}

			ImGui::Text("Preview quality: %d", current_state ? int(current_state->quality) : 0);
		}

		static double tmin = 0.0;
		static double tmax = 1.0;
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
		if (ImGui::SliderScalar("##Time", ImGuiDataType_Double, &current_time, &tmin, &tmax)) {
			render_options_changed = true;
		}
		ImGui::PopItemWidth();

		auto avail = ImGui::GetContentRegionAvail();

		if (avail.x < 0) avail.x = 10;
		if (avail.y < 0) avail.y = 10;

		preview_size = { static_cast<unsigned int>(avail.x), static_cast<unsigned int>(avail.y) };

		if (displayed_texture.has_value()) {
			auto& tex = displayed_texture.value();
			ImGui::Image((void*)(intptr_t)tex.id(), ImVec2(preview_size.x, preview_size.y));
			preview_hovered = ImGui::IsItemHovered();
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

			cmd_exec({
				[&flame, new_value = double2(flame.center_x.t0, flame.center_y.t0)]() mutable {
					flame.center_x.t0 = new_value.x;
					flame.center_y.t0 = new_value.y;
				},
				[&flame, old_value = drag_start]() mutable {
					flame.center_x.t0 = old_value.x;
					flame.center_y.t0 = old_value.y;
				}
			});
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

rfkt::hash_t preview_panel::get_value_hash(const rfkt::flame& flame)
{
	rfkt::hash::state_t state;

	auto process = [&state](const rfkt::anima& v) mutable {
		state.update(v.t0);

		if (v.call_info) {
			state.update(v.call_info->first);
			for (const auto& [name, value] : v.call_info->second) {
				state.update(name);
				std::visit([&](auto argv) {
					state.update(argv);
				}, value);
			}
		}
	};

	process(flame.center_x);
	process(flame.center_y);
	process(flame.scale);
	process(flame.rotate);

	process(flame.gamma);
	process(flame.brightness);
	process(flame.vibrancy);

	flame.for_each_xform([&](int xid, const rfkt::xform& xf) {
		process(xf.weight);
		process(xf.color);
		process(xf.color_speed);
		process(xf.opacity);

		for (const auto& vl : xf.vchain) {
			vl.transform.pack(process);
			process(vl.mod_rotate);
			process(vl.mod_scale);
			process(vl.mod_x);
			process(vl.mod_y);

			for (const auto& [vname, vd] : vl) {
				process(vd.weight);

				for (const auto& [pname, val] : vd) {
					process(val);
				}
			}
		}
	});

	state.update(flame.palette);

	return state.digest();
}
