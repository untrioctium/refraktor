#include <pugixml.hpp>
#include <ranges>
#include <array>
#include <charconv>
#include <spdlog/spdlog.h>

#include "librefrakt/flame_info.h"
#include "librefrakt/flame_types.h"
#include "librefrakt/util/color.h"

std::vector<double> string_to_doubles(std::string_view s) {
	std::vector<double> ret{};
	for (double v : s
		| std::views::split(std::string_view{ " " })
		| std::views::transform(
			[](auto sv) {
				double v = 0.0;
				std::from_chars(sv.data(), sv.data() + sv.size(), v);
				return v;
			}
		)
	) 
	{
		ret.push_back(v);
	}

	return ret;
}

rfkt::xform from_flam3_xml(const rfkt::flamedb& fdb, const pugi::xml_node& node) noexcept
{
	auto xf = rfkt::xform{};
	auto vlinks = std::array<rfkt::vlink, 3>{};
	auto has_post_affine = false;

	auto get_variation_name = [&fdb](std::string_view param_name) -> std::pair<std::string_view, std::string_view> {
		std::string_view match = {};
		for (const auto& v : fdb.variations()) {
			if (param_name.starts_with(v.name) && param_name.size() > match.size()) {
				match = v.name;
			}
		}

		if (!match.empty()) {
			return { match, param_name.substr(match.size() + 1) };
		}

		return {};
	};
	 
	for (const auto& attr : node.attributes()) {
		std::string_view aname = attr.name();
		auto which_vl = 1;

		if (aname.starts_with("pre_")) {
			which_vl = 0;
			aname = aname.substr(4);
			if (aname == "blur") aname = "gaussian_blur";
		}
		else if (aname.starts_with("post_")) {
			which_vl = 2;
			aname = aname.substr();
		}

		auto& cur_vl = vlinks[which_vl];

		if (aname == "weight") xf.weight = attr.as_double();
		else if (aname == "color") xf.color = attr.as_double();
		else if (aname == "color_speed") xf.color_speed = attr.as_double();
		else if (aname == "opacity") xf.opacity = attr.as_double();
		else if (aname == "coefs" || aname == "post") {
			if (aname == "post") has_post_affine = true;

			auto vec = string_to_doubles(attr.value());
			while (vec.size() < 6) vec.push_back(0.0);

			vlinks[(aname == "coefs")? 1: 2].transform = rfkt::affine{ vec[0], vec[1], vec[2], vec[3], vec[4], vec[5] };
		}
		else if (fdb.is_variation(aname)) {
			if (!cur_vl.has_variation(aname)) {
				cur_vl.add_variation(fdb.make_vardata(aname));
			}
			cur_vl[aname].weight = attr.as_double();
		}
		else if (auto [vname, pname] = get_variation_name(aname); !vname.empty()) {
			if (!cur_vl.has_variation(vname)) {
				cur_vl.add_variation(fdb.make_vardata(vname));
			}
			cur_vl[vname][pname] = attr.as_double();
		}
		else if(aname != "animate") {
			SPDLOG_WARN("Unknown xform attribute: {}", aname);
		}
	}

	if (vlinks[0].size_variations() > 0) {
		xf.vchain.emplace_back(std::move(vlinks[0]));
	}

	xf.vchain.emplace_back(std::move(vlinks[1]));

	if (vlinks[2].size_variations() > 0 || has_post_affine) {
		if (vlinks[2].size_variations() == 0) {
			vlinks[2].add_variation(fdb.make_vardata("linear"));
			vlinks[2]["linear"].weight = 1.0;
		}
		xf.vchain.emplace_back(std::move(vlinks[2]));
	}

	return xf;
}

auto rfkt::import_flam3(const flamedb& fdb, std::string_view content) noexcept -> std::optional<flame>
{
	auto doc = pugi::xml_document();

	if (auto result = doc.load_string(content.data()); !result) {
		return std::nullopt;
	}

	auto ret = flame{};

	auto flame_node = doc.child("flame");

	auto size = string_to_doubles(flame_node.attribute("size").value());

	ret.scale = flame_node.attribute("scale").as_double() / size[1];
	ret.rotate = flame_node.attribute("rotate").as_double();
	
	auto center = string_to_doubles(flame_node.attribute("center").value());
	if (center.size() < 2) center.resize(2, 0.0);
	ret.center_x = center[0];
	ret.center_y = center[1];

	ret.gamma = flame_node.attribute("gamma").as_double();
	ret.brightness = flame_node.attribute("brightness").as_double();
	ret.vibrancy = flame_node.attribute("vibrancy").as_double();

	for (const auto& node : flame_node.children("xform")) {
		ret.xforms.emplace_back(from_flam3_xml(fdb, node));
	}

	if(auto fnode = flame_node.child("finalxform"); fnode) {
		ret.final_xform = from_flam3_xml(fdb, fnode); 
	}

	auto colors = flame_node.children("color");
	auto color_count = 0;
	for (const auto& _ : colors) {
		color_count++;
	}

	ret.palette.resize(color_count);
	for (const auto& color : colors) {
		auto index = color.attribute("index").as_ullong();
		auto rgb = string_to_doubles(color.attribute("rgb").value());
		if (rgb.size() < 3) rgb.resize(3, 0.0);
		auto hsv = rfkt::color::rgb_to_hsv({ rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0 });
		ret.palette[index] = { hsv.x, hsv.y, hsv.z };
	}

	return ret;
}
