#include <pugixml.hpp>
#include <ranges>
#include <array>
#include <charconv>
#include <spdlog/spdlog.h>

#include "librefrakt/flame_info.h"
#include "librefrakt/flame_types.h"
#include "librefrakt/util/color.h"
#include "librefrakt/anima.h"
#include "librefrakt/util/zlib.h"

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
		else if (aname == "animate") {
			if (attr.as_double() == 1) vlinks[1].mod_rotate.call_info = { "increase", {{"per_loop", 360.0}} };
		}
		else {
			SPDLOG_WARN("Unknown xform attribute: {}", aname);
		}
	}

	if (vlinks[0].size_variations() > 0) {
		vlinks[0].transform = rfkt::affine::identity();
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

	ret.name = flame_node.attribute("name").value();
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

#define CHECK_AND_DESERIALIZE_ANIMA(target, field) \
	if(!js.contains(#field)) return std::nullopt; \
	auto field ## _opt = anima::deserialize(js[#field], ft); \
	if (!field ## _opt) return std::nullopt; \
	target.field = std::move(*field ## _opt)

ordered_json rfkt::anima::serialize() const noexcept {
	if (!call_info) return t0;
	auto result = ordered_json::object();
	result["t0"] = t0;

	result["call"] = call_info->first;
	result["args"] = ordered_json::object();
	for (auto& [k, v] : call_info->second) {
		if (std::holds_alternative<int>(v)) {
			result["args"][k] = std::get<int>(v);
		}
		else if (std::holds_alternative<double>(v)) {
			result["args"][k] = std::get<double>(v);
		}
		else if (std::holds_alternative<bool>(v)) {
			result["args"][k] = std::get<bool>(v);
		}
	}

	return result;
}

std::optional<rfkt::anima> rfkt::anima::deserialize(const json& js, const function_table& ft) noexcept {
	if (js.is_number()) return js.get<double>();

	if (!js.is_object()) return std::nullopt;
	if (!js.contains("t0") || !js["t0"].is_number()) return std::nullopt;

	auto t0 = js["t0"].get<double>();
	if (js.contains("call")) {
		auto call = js["call"].get<std::string>();
		auto args = js["args"].get<json::object_t>();
		auto arg_map = arg_map_t();
		for (auto& [k, v] : args) {
			if (v.is_number_integer()) {
				arg_map[k] = v.get<int>();
			}
			else if (v.is_number_float()) {
				arg_map[k] = v.get<double>();
			}
			else if (v.is_boolean()) {
				arg_map[k] = v.get<bool>();
			}
		}

		return anima(t0, std::make_pair(call, arg_map));
	}
	else {
		return anima(t0);
	}
}

std::optional<rfkt::vardata> rfkt::vardata::deserialize(std::string_view name, const json& js, const function_table& ft, const flamedb& fdb)
{
	if (!fdb.is_variation(name)) return std::nullopt;

	auto [_, vdata] = fdb.make_vardata(name);
	
	// variations with only weight
	if (js.is_number()) {
		if (vdata.parameters_.empty()) {
			vdata.weight = js.get<double>();
			return vdata;
		}
		else return std::nullopt;
	}

	// variations with only an animated weight
	if (js.is_object() && js.contains("t0")) {
		if (vdata.parameters_.empty()) {
			auto a = anima::deserialize(js, ft);
			if (!a) return std::nullopt;
			vdata.weight = std::move(*a);
			return vdata;
		}
		else return std::nullopt;
	}

	// anything past this point needs an object 
	if (!js.is_object()) return std::nullopt;

	if (!js.contains("weight")) return std::nullopt;
	if (auto weight = rfkt::anima::deserialize(js["weight"], ft); weight) {
		vdata.weight = std::move(*weight);
	}
	else return std::nullopt;

	if (!js.contains("parameters")) return vdata;
	if (!js["parameters"].is_object()) return std::nullopt;

	for (const auto& item : js["parameters"].items()) {
		if (!vdata.parameters_.contains(item.key())) return std::nullopt;

		auto a = anima::deserialize(item.value(), ft);
		if (!a) return std::nullopt;

		vdata.parameters_[item.key()] = std::move(*a);
	}

	return vdata;
}

std::optional<rfkt::vlink> rfkt::vlink::deserialize(const json& js, const function_table& ft, const flamedb& fdb)
{
	if (!js.is_object()) return std::nullopt;

	auto ret = rfkt::vlink{};

	if (!js.contains("transform")) return std::nullopt;
	auto transform = affine::deserialize(js["transform"], ft);
	if (!transform) return std::nullopt;
	ret.transform = std::move(*transform);

	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_x);
	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_y);
	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_scale);
	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_rotate);

	if (!js.contains("variations") || !js["variations"].is_object()) return std::nullopt;

	for (const auto& var : js["variations"].items()) {
		auto vdata_opt = vardata::deserialize(var.key(), var.value(), ft, fdb);
		if (!vdata_opt) return std::nullopt;

		ret.variations_.insert_or_assign(var.key(), std::move(*vdata_opt));
	}

	return ret;
}

std::optional<rfkt::xform> rfkt::xform::deserialize(const json& js, const function_table& ft, const flamedb& fdb) {
	if (!js.is_object()) return std::nullopt;

	auto ret = rfkt::xform{};

	CHECK_AND_DESERIALIZE_ANIMA(ret, weight);
	CHECK_AND_DESERIALIZE_ANIMA(ret, color);
	CHECK_AND_DESERIALIZE_ANIMA(ret, color_speed);
	CHECK_AND_DESERIALIZE_ANIMA(ret, opacity);

	if (!js.contains("vchain") || !js["vchain"].is_array()) return std::nullopt;

	for (const auto& link : js["vchain"]) {
		auto link_opt = vlink::deserialize(link, ft, fdb);
		if (!link_opt) return std::nullopt;

		ret.vchain.push_back(std::move(*link_opt));
	}

	return ret;
}

ordered_json rfkt::flame::serialize() const noexcept {
	ordered_json js;

	js["center_x"] = center_x.serialize();
	js["center_y"] = center_y.serialize();
	js["scale"] = scale.serialize();
	js["rotate"] = rotate.serialize();

	js["gamma"] = gamma.serialize();
	js["brightness"] = brightness.serialize();
	js["vibrancy"] = vibrancy.serialize();

	js["xforms"] = ordered_json::array();

	for (const auto& xf : xforms) {
		js["xforms"].emplace_back(xf.serialize());
	}

	if (final_xform) {
		js["final_xform"] = final_xform->serialize();
	}



	constexpr static auto palette_element_size = sizeof(decltype(palette)::value_type);
	auto total_size_bytes = palette.size() * palette_element_size;

	js["palette"] = zlib::compress_b64(palette.data(), total_size_bytes);

	return js;
}

std::optional<rfkt::flame> rfkt::flame::deserialize(const json& js, const function_table& ft, const flamedb& fdb) {
	if (!js.is_object()) return std::nullopt;

	auto ret = rfkt::flame{};

	CHECK_AND_DESERIALIZE_ANIMA(ret, center_x);
	CHECK_AND_DESERIALIZE_ANIMA(ret, center_y);
	CHECK_AND_DESERIALIZE_ANIMA(ret, scale);
	CHECK_AND_DESERIALIZE_ANIMA(ret, rotate);

	CHECK_AND_DESERIALIZE_ANIMA(ret, gamma);
	CHECK_AND_DESERIALIZE_ANIMA(ret, brightness);
	CHECK_AND_DESERIALIZE_ANIMA(ret, vibrancy);

	if (!js.contains("xforms") || !js.is_array()) return std::nullopt;

	for (const auto& xf : js["xforms"]) {
		auto xf_opt = xform::deserialize(xf, ft, fdb);
		if (!xf_opt) return std::nullopt;
		ret.xforms.emplace_back(std::move(*xf_opt));
	}

	if (js.contains("final_xform")) {
		auto fxf_opt = xform::deserialize(js["final_xform"], ft, fdb);
		if (!fxf_opt) return std::nullopt;
		ret.final_xform = fxf_opt;
	}

	if (!js.contains("palette") || !js["palette"].is_string()) return std::nullopt;

	auto palette_data = zlib::uncompress_b64(js["palette"]);

	constexpr static auto palette_element_size = sizeof(decltype(palette)::value_type);
	if (palette_data.size() % palette_element_size != 0) return std::nullopt;

	auto total_elements = palette_data.size() / palette_element_size;
	ret.palette.resize(total_elements);

	// TODO: use a less evil way not full of potential UB
	std::memcpy(ret.palette.data(), palette_data.data(), palette_data.size());

	return ret;
}

rfkt::hash_t rfkt::flame::value_hash() const noexcept
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

	process(center_x);
	process(center_y);
	process(scale);
	process(rotate);

	process(gamma);
	process(brightness);
	process(vibrancy);

	for_each_xform([&](int xid, const rfkt::xform& xf) {
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

	state.update(palette);

	return state.digest();
}
