#include "gui/panels/preview_panel.h"

std::vector<rfkt::flame> gen_samples2(const rfkt::flame& f, double t, double loops_per_frame) {
	std::vector<rfkt::flame> samples{};
	for (int i = 0; i < 4; i++) {
		auto sample = f;
		for (auto& x : sample.xforms) {
			for (auto& vl : x.vchain) {
				if (vl.per_loop > 0) {
					const auto base = t * vl.per_loop;
					vl.transform = vl.transform.rotated(base + vl.per_loop * loops_per_frame * (i - 1));
				}
			}
		}
		samples.emplace_back(std::move(sample));
	}
	return samples;
}

preview_panel::~preview_panel()
{
	if (rendering_texture) {
		auto tex = rendering_texture->get();
		cuda_map = std::nullopt;
	}
}

bool preview_panel::show(const rfkt::flamedb& fdb, rfkt::flame& flame)
{

	if (render_is_ready()) {
		displayed_texture = rendering_texture->get();
		cuda_map = std::nullopt;
		rendering_texture = std::nullopt;
	}

	auto preview_size = gui_logic(flame);

	static bool first_frame = false;
	if (first_frame) {
		preview_size = { 10, 10 };
		first_frame = false;
	}

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
			|| (displayed_texture && (preview_size.x != render_dims.x || preview_size.y != render_dims.y));

		bool needs_render = needs_clear || (current_state && current_state->quality < target_quality);

		if (needs_render) {
			auto out_tex = rfkt::gl::texture{ preview_size.x, preview_size.y };
			cuda_map = out_tex.map_to_cuda();

			std::vector<rfkt::flame> samples = needs_clear ? gen_samples2(flame, current_time, 1.0 / 300) : std::vector<rfkt::flame>{};

			auto promise = std::promise<rfkt::gl::texture>();
			auto future = promise.get_future();

			submitter([promise = std::move(promise),
				&renderer = this->renderer,
				out_tex = std::move(out_tex), &cuda_map = this->cuda_map.value(),
				needs_kernel, needs_clear,
				samples = std::move(samples),
				&compiler = this->compiler, kernel = this->kernel, &fdb,
				state = this->current_state,
				gbv = double3{ flame.gamma, flame.brightness, flame.vibrancy },
				&stream = this->stream, preview_size,
				target_quality = this->target_quality,
				upscale = this->upscale]() mutable {

					if (needs_kernel) {
						auto result = compiler.get_flame_kernel(fdb, rfkt::precision::f32, samples.at(0));
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
	if (ImGui::Begin("Render", nullptr, ImGuiWindowFlags_MenuBar)) {

		auto avail_before = ImGui::GetContentRegionAvail();
		if (ImGui::BeginMenuBar()) {
			if (ImGui::BeginMenu("Render options")) {
				if (ImGui::MenuItem("Upscale 2x", nullptr, &this->upscale)) { 
					render_options_changed = true;
				}
				rfkt::gui::tooltip("Enables upscaling; improves performance but reduces quality.", false);
				ImGui::EndMenu();
			}

			const static std::vector<std::pair<std::string, double>> qualities {
				{"Low", 32},
				{"Medium", 128},
				{"High", 512},
				{"Ultra", 2048},
				{"Unlimited", std::pow(2.0, 31.0)}
			};

			if (ImGui::BeginMenu("Quality")) {

				for (const auto& [name, quality]: qualities) {
					if (ImGui::MenuItem(name.c_str())) {
						target_quality = quality;
						render_options_changed = true;
					}
				}
				ImGui::EndMenu();
			}

			ImGui::Text("Preview quality: %d", current_state ? int(current_state->quality) : 0);

			ImGui::EndMenuBar();
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
	ImGui::End();

	if (!dragging && preview_hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
		dragging = true;
		drag_start = { flame.center_x, flame.center_y };
		last_delta = ImGui::GetMouseDragDelta();
	}
	else if (dragging) {
		if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
			dragging = false;

			cmd_exec({
				[&flame, new_value = double2(flame.center_x, flame.center_y)]() mutable {
					flame.center_x = new_value.x;
					flame.center_y = new_value.y;
				},
				[&flame, old_value = drag_start]() mutable {
					flame.center_x = old_value.x;
					flame.center_y = old_value.y;
				}
			});
		}
		else {
			auto delta = ImGui::GetMouseDragDelta();
			auto drag_dist = delta - last_delta;
			if (drag_dist.x != 0 || drag_dist.y != 0) {
				auto scale = flame.scale;
				auto rot = -flame.rotate * IM_PI / 180;

				double2 vector = { drag_dist.x / (scale * preview_size.y) , drag_dist.y / (scale * preview_size.y) };
				double2 rotated = { vector.x * cos(rot) - vector.y * sin(rot), vector.x * sin(rot) + vector.y * cos(rot) };

				flame.center_x -= rotated.x;
				flame.center_y -= rotated.y;
			}

			last_delta = delta;
		}
	}

	if (preview_hovered) {
		ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
	}

	if (auto scroll = ImGui::GetIO().MouseWheel; scroll != 0 && preview_hovered && !dragging) {
		flame.scale *= std::pow(1.1, scroll);
	}

	return preview_size;
}

rfkt::hash_t preview_panel::get_value_hash(const rfkt::flame& flame)
{
	const auto buffer_size = 7 + flame.size_reals();
	auto buffer = std::vector<double>(buffer_size);

	auto packer = [buf_view = std::span{ buffer }, counter = 0](double v) mutable {
		buf_view[counter] = v;
		counter++;
	};

	packer(flame.center_x);
	packer(flame.center_y);
	packer(flame.scale);
	packer(flame.rotate);

	packer(flame.gamma);
	packer(flame.brightness);
	packer(flame.vibrancy);

	flame.pack(packer);

	rfkt::hash::state_t state;

	state.update(buffer);
	state.update(flame.palette);

	return state.digest();
}
