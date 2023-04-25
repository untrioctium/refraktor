#include "gui/panels/preview_panel.h"

bool preview_panel::show(rfkt::flame& flame)
{
	rfkt::gui::scope::style padding_scope = { ImGuiStyleVar_WindowPadding, ImVec2(0, 0) };

	//if (ImGui::Begin("Render")) {
	//	auto avail = ImGui::GetContentRegionAvail();

	//	return { static_cast<unsigned int>(avail.x), static_cast<unsigned int>(avail.y) };
	//}
	ImGui::End();

	if (render_is_ready()) {
		displayed_texture = std::move(rendering_texture->get());
		rendering_texture.reset();
	}
	else if (!rendering_texture) {

		const auto given_struct_hash = flame.hash();
		const auto given_value_hash = get_value_hash(flame);

		bool needs_render = (given_struct_hash != flame_structure_hash) || (given_value_hash != flame_value_hash);

	}



	return false;
}


rfkt::hash_t preview_panel::get_value_hash(const rfkt::flame& flame)
{
	const auto buffer_size = 4 + flame.size_reals();
	auto buffer = std::vector<double>(buffer_size);

	auto packer = [buf_view = std::span{ buffer }, counter = 0](double v) mutable {
		buf_view[counter] = v;
		counter++;
	};

	packer(flame.center_x);
	packer(flame.center_y);
	packer(flame.scale);
	packer(flame.rotate);

	flame.pack(packer);

	rfkt::hash::state_t state;

	state.update(buffer.data(), buffer.size());

	const auto palette_size_bytes = sizeof(decltype(flame.palette)::value_type) * flame.palette.size();

	state.update(flame.palette.data(), palette_size_bytes);

	return state.digest();
}
