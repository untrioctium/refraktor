#include <pugixml.hpp>
#include <ranges>
#include <array>
#include <charconv>
#include <cmath>
#include <set>
#include <algorithm>
#include <string_view>
#include <spdlog/spdlog.h>
#include <sol/sol.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include "librefrakt/flame_info.hpp"
#include "librefrakt/flame_types.hpp"
#include "librefrakt/util/color.hpp"
#include "librefrakt/anima.hpp"
#include "librefrakt/util/zlib.hpp"

#include <dlib/optimization/max_cost_assignment.h>

std::vector<double> string_to_doubles(std::string_view s) {
	std::vector<double> ret{};
	for (double v : s
		| std::views::split(std::string_view{ " " })
		| std::views::transform(
			[](auto sv) {
				double v = 0.0;
				std::from_chars(sv.data(), sv.data() + sv.size(), v); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				return v;
			}
		)
	) 
	{
		ret.push_back(v);
	}

	return ret;
}

std::expected<rfkt::xform, std::string> from_flam3_xml(const rfkt::flamedb& fdb, const pugi::xml_node& node) noexcept
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
			if (aname == "blur") {
				aname = "gaussian_blur";
				if(!vlinks[0].has_variation("linear")) {
					vlinks[0].add_variation(fdb.make_vardata("linear"));
				}

				vlinks[0]["linear"].weight.t0 += 1.0;
			}
		}
		else if (aname.starts_with("post_")) {
			which_vl = 2;
			aname = aname.substr();
		}

		auto& cur_vl = vlinks.at(which_vl);

		if (aname == "weight") xf.weight = attr.as_double();
		else if (aname == "color") xf.color = attr.as_double();
		else if (aname == "color_speed") xf.color_speed = attr.as_double();
		else if (aname == "opacity") xf.opacity = attr.as_double();
		else if (aname == "symmetry") {
			xf.color_speed = (1.0 - attr.as_double()) / 2.0;
			if(attr.as_double() <= 0) {
				vlinks[1].mod_rotate.call_info = { "increase", {{"per_loop", 360.0}} };
			}
		}
		else if (aname == "coefs" || aname == "post") {
			if (aname == "post") has_post_affine = true;

			auto vec = string_to_doubles(attr.value());
			while (vec.size() < rfkt::affine::size_reals()) vec.push_back(0.0);

			vlinks.at((aname == "coefs")? 1: 2).transform = rfkt::affine{ vec[0], vec[1], vec[2], vec[3], vec[4], vec[5] };
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
			if(!fdb.is_parameter(vname, pname)) {
				return std::unexpected(std::format("Unknown xform parameter: {}/{}", vname, pname));
			}
			cur_vl[vname][pname] = attr.as_double();
		}
		else if (aname == "animate") {
			if (attr.as_double() == 1) vlinks[1].mod_rotate.call_info = { "increase", {{"per_loop", 360.0}} };
		}
		else if (aname != "chaos") {
			return std::unexpected(std::format("Unknown xform attribute: {}", aname));
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

	return std::expected<rfkt::xform, std::string>{std::in_place, std::move(xf)};
}

constexpr bool is_hex(char value) {
	return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F');
}

constexpr unsigned char hex_to_int(char value) {
	if (value >= '0' && value <= '9') return value - '0';
	if (value >= 'a' && value <= 'f') return value - 'a' + 10;
	if (value >= 'A' && value <= 'F') return value - 'A' + 10;
	return 0;
}

auto rfkt::import_flam3(const flamedb& fdb, std::string_view content) noexcept -> std::expected<flame, std::string>
{
	auto doc = pugi::xml_document();

	if (auto result = doc.load_string(content.data()); !result) {
		return std::unexpected(result.description());
	}

	auto ret = flame{};

	auto flame_node = [&]() {
		if (doc.first_child().name() == std::string_view{ "flame" })
			return doc.first_child();
		else
			return doc.child("flames").child("flame");
	}();

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
	ret.highlight_power = flame_node.attribute("highlight_power").as_double();
	ret.gamma_threshold = flame_node.attribute("gamma_threshold").as_double();

	std::map<int, std::string> chaos_table{};

	int xid = 0;
	for (const auto& node : flame_node.children("xform")) {

		if (auto chaos = node.attribute("chaos"); chaos) {
			chaos_table[xid] = chaos.as_string();
		}
		xid++;
		if(auto xf = from_flam3_xml(fdb, node); !xf) {
			return std::unexpected(std::format("Could not parse xform {}: {}", xid, xf.error()));
		} else {
			ret.add_xform(std::move(*xf));
		}
	}

	if(auto fnode = flame_node.child("finalxform"); fnode) {
		if(auto xf = from_flam3_xml(fdb, fnode); !xf) {
			return std::unexpected(std::format("Could not parse final xform: {}", xf.error()));
		} else {
			ret.final_xform = std::move(*xf);
		}
	}

	if (auto pnode = flame_node.child("palette"); pnode) {
		auto count = pnode.attribute("count").as_ullong();

		ret.palette.resize(count);

		auto pal_data = std::string_view{pnode.text().as_string()};
		int current_index = 0;
		std::vector<char> hex_color;
		for (int idx = 0; pal_data[idx] != '\0'; idx++) {
			auto ch = pal_data[idx];
			if (!is_hex(ch)) continue;

			hex_color.push_back(ch);

			if (hex_color.size() == 6) {
				auto rgb = std::array<double, 3>{};
				rgb[0] = hex_to_int(hex_color[0]) * 16 + hex_to_int(hex_color[1]);
				rgb[1] = hex_to_int(hex_color[2]) * 16 + hex_to_int(hex_color[3]);
				rgb[2] = hex_to_int(hex_color[4]) * 16 + hex_to_int(hex_color[5]);

				auto hsv = rfkt::color::rgb_to_hsv({ rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0 });
				ret.palette[current_index] = { hsv.x, hsv.y, hsv.z };

				hex_color.clear();
				current_index++;
			}
		}
	}
	else {
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
			auto hsv = rfkt::color::rgb_to_hsv({ rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0 });
			ret.palette[index] = { hsv.x, hsv.y, hsv.z };
		}
	}

	if (!chaos_table.empty()) {
		ret.add_chaos();
		for (const auto& [idx, chaos] : chaos_table) {
			auto vals = string_to_doubles(chaos);

			for (int j = 0; j < ret.xforms().size(); j++) {
				ret.chaos_table.value()[idx][j].t0 = (j >= vals.size())? 1.0: vals[j];
			}
		}
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

	result["call"] = call_info->name;
	result["args"] = ordered_json::object();
	for (auto& [k, v] : call_info->args) {
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

		return anima(t0, call_info_value_t{ call, arg_map });
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
	js["highlight_power"] = highlight_power.serialize();
	js["gamma_threshold"] = gamma_threshold.serialize();

	js["mod_hue"] = mod_hue.serialize();
	js["mod_sat"] = mod_sat.serialize();
	js["mod_val"] = mod_val.serialize();

	js["xforms"] = ordered_json::array();

	for (const auto& xf : xforms_) {
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
	CHECK_AND_DESERIALIZE_ANIMA(ret, highlight_power);
	CHECK_AND_DESERIALIZE_ANIMA(ret, gamma_threshold);

	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_hue);
	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_sat);
	CHECK_AND_DESERIALIZE_ANIMA(ret, mod_val);

	if (!js.contains("xforms") || !js["xforms"].is_array()) return std::nullopt;

	for (const auto& xf : js["xforms"]) {
		auto xf_opt = xform::deserialize(xf, ft, fdb);
		if (!xf_opt) return std::nullopt;
		ret.add_xform(std::move(*xf_opt));
	}

	if (js.contains("final_xform")) {
		auto fxf_opt = xform::deserialize(js["final_xform"], ft, fdb);
		if (!fxf_opt) return std::nullopt;
		ret.final_xform = fxf_opt;
	}

	if (!js.contains("palette") || !js["palette"].is_string()) return std::nullopt;

	auto palette_data = zlib::uncompress_b64(js["palette"].get<std::string>());

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
			state.update(v.call_info->name);
			for (const auto& [name, value] : v.call_info->args) {
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
	process(highlight_power);
	process(gamma_threshold);

	process(mod_hue);
	process(mod_sat);
	process(mod_val);

	if (chaos_table.has_value()) {
		for (const auto& row : chaos_table.value()) {
			for(const auto& val : row) {
				process(val);
			}
		}
	}

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

rfkt::anima rfkt::anima::interpolate(anima o, double start_time, double length) const {
	auto new_anima = anima{ t0 };
	new_anima.call_info = call_info_value_t{};
	auto& nargs = new_anima.call_info->args;

	if (call_info) {
		nargs["left.function"] = call_info->name;
		for (const auto& [name, value] : call_info->args) {
			nargs["left." + name] = value;
		}
	}

	nargs["right.t0"] = o.t0;
	if (o.call_info) {
		nargs["right.function"] = o.call_info->name;
		for (const auto& [name, value] : o.call_info->args) {
			nargs["right." + name] = value;
		}
	}

	nargs["start_time"] = start_time;
	nargs["length"] = length;

	return new_anima;
}

rfkt::affine rfkt::affine::rotated(double deg) const noexcept {
	double rad = -glm::radians(deg);

	glm::dmat4 m = {
		a.t0, b.t0, 0, 0,
		d.t0, e.t0, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	auto newmat = glm::rotate(m, rad, glm::dvec3(0, 0, 1));

	return {
		newmat[0][0], newmat[1][0], newmat[0][1], newmat[1][1], c.t0, f.t0
	};
}

ordered_json rfkt::affine::serialize() const noexcept {
	return ordered_json::array({ a.serialize(), d.serialize(), b.serialize(), e.serialize(), c.serialize(), f.serialize() });
}

std::optional<rfkt::affine> rfkt::affine::deserialize(const json& js, const function_table& ft) noexcept {
	if (!js.is_array()) return std::nullopt;

	auto arr = js.get<json::array_t>();
	if (arr.size() != 6) return std::nullopt;

	auto a = anima::deserialize(arr[0], ft);
	auto d = anima::deserialize(arr[1], ft);
	auto b = anima::deserialize(arr[2], ft);
	auto e = anima::deserialize(arr[3], ft);
	auto c = anima::deserialize(arr[4], ft);
	auto f = anima::deserialize(arr[5], ft);

	if (!a || !b || !c || !d || !e || !f) return std::nullopt;

	return affine{ std::move(*a), std::move(*d), std::move(*b), std::move(*e), std::move(*c), std::move(*f) };
}

double rfkt::affine::distance(const rfkt::affine& o) const noexcept {
	double dist = 0.0;
	dist += std::pow(a.t0 - o.a.t0, 2);
	dist += std::pow(b.t0 - o.b.t0, 2);
	dist += std::pow(c.t0 - o.c.t0, 2);
	dist += std::pow(d.t0 - o.d.t0, 2);
	dist += std::pow(e.t0 - o.e.t0, 2);
	dist += std::pow(f.t0 - o.f.t0, 2);
	return std::sqrt(dist);
}

ordered_json rfkt::vardata::serialize() const noexcept {
	if (parameters_.empty()) return weight.serialize();

	ordered_json js;
	js["weight"] = weight.serialize();
	js["parameters"] = json::object();

	for (const auto& [name, value] : parameters_) {
		js["parameters"][name] = value.serialize();
	}

	return js;
}

rfkt::anima* rfkt::vardata::lookup(std::string_view path) {
	auto [head, tail] = detail::split_path(path);
	if (head == "weight") return &weight;
	if (head == "parameter") {
		auto [param_name, _] = detail::split_path(tail);
		if(param_name.empty()) return nullptr;
		auto iter = parameters_.find(param_name);
		if(iter == parameters_.end()) return nullptr;
		return &iter->second;
	}
	return nullptr;
}

void rfkt::vlink::add_to_hash(rfkt::hash::state_t& hs) const {
	for (const auto& [name, _] : variations_) {
		hs.update(name);
	}
}

ordered_json rfkt::vlink::serialize() const noexcept {
	ordered_json js;
	js["transform"] = transform.serialize();
	js["mod_x"] = mod_x.serialize();
	js["mod_y"] = mod_y.serialize();
	js["mod_scale"] = mod_scale.serialize();
	js["mod_rotate"] = mod_rotate.serialize();
	js["variations"] = ordered_json::object();

	for (const auto& [name, value] : variations_) {
		js["variations"][name] = value.serialize();
	}

	return js;
}

rfkt::vlink rfkt::vlink::identity() {
	auto vl = vlink{};
	vl.transform = affine::identity();
	vl.add_variation(vardata::identity());
	return vl;
}

rfkt::anima* rfkt::vlink::lookup(std::string_view path) {
	auto [head, tail] = detail::split_path(path);
	if (head == "transform") return transform.lookup(tail);
	if (auto ptr = name_to_pointer(head); ptr) return &(this->*ptr);
	if (head == "variation") {
		auto [var_name, _] = detail::split_path(tail);
		if(var_name.empty()) return nullptr;
		auto iter = variations_.find(var_name);
		if(iter == variations_.end()) return nullptr;
		return iter->second.lookup(tail);
	}
	return nullptr;
}

double rfkt::vlink::similarity(const rfkt::vlink* o) const noexcept {
	std::set<std::string_view> vars_a{};
	std::set<std::string_view> vars_b{};

	for(const auto& [name, data] : variations_) {
		vars_a.insert(name);
	}

	for(const auto& [name, data] : o->variations_) {
		vars_b.insert(name);
	}

	std::set<std::string_view> intersection {};
	std::set_intersection(vars_a.begin(), vars_a.end(), vars_b.begin(), vars_b.end(), std::inserter(intersection, intersection.begin()));

	auto union_size = vars_a.size() + vars_b.size() - intersection.size();
	double jaccard_index = static_cast<double>(intersection.size()) / static_cast<double>(union_size);

	auto sum_weights = [](const rfkt::vlink& v) {
		double total = 0.0;
		for(const auto& [name, data] : v.variations_) {
			total += std::abs(data.weight.t0);
		}
		return total;
	};

	auto total_weight_a = sum_weights(*this);
	auto total_weight_b = sum_weights(*o);

	double weight_sim = 0;
	for(const auto name : intersection) {
		double wa = variations_.find(name)->second.weight.t0 / total_weight_a;
		double wb = o->variations_.find(name)->second.weight.t0 / total_weight_b;

		weight_sim += 1.0 - std::abs(wa - wb) / std::max({wa, wb, 1e-6});
	}
	if(!intersection.empty()) weight_sim /= static_cast<double>(intersection.size());

	return 0.5 * jaccard_index + 0.5 * weight_sim;
}

void rfkt::xform::add_to_hash(rfkt::hash::state_t& hs) const {
	for (std::size_t i = 0; i < vchain.size(); i++) {
		hs.update(0xBULL); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
		vchain.at(i).add_to_hash(hs);
	}
}

ordered_json rfkt::xform::serialize() const noexcept {
	ordered_json js;
	js["weight"] = weight.serialize();
	js["color"] = color.serialize();
	js["color_speed"] = color_speed.serialize();
	js["opacity"] = opacity.serialize();
	js["vchain"] = ordered_json::array();

	for (const auto& link : vchain) {
		js["vchain"].emplace_back(link.serialize());
	}

	return js;
}

rfkt::xform rfkt::xform::identity() {
	auto xf = xform{};
	xf.weight = 0.0;
	xf.color = 0.0;
	xf.color_speed = 0.0;
	xf.opacity = 1.0;
	xf.vchain.emplace_back(vlink::identity());
	return xf;
}

rfkt::anima* rfkt::xform::lookup(std::string_view path) {
	auto [head, tail] = detail::split_path(path);
	if (auto ptr = name_to_pointer(head); ptr) return &(this->*ptr);
	if (head == "vlink") {
		auto [vlink_idx, vlink_path] = detail::split_path(tail);
		if(vlink_idx.empty()) return nullptr;
		int idx = std::stoi(std::string(vlink_idx));
		if(idx < 0 || idx >= vchain.size()) return nullptr;
		return vchain[idx].lookup(vlink_path);
	}
	return nullptr;
}

void rfkt::flame::add_to_hash(rfkt::hash::state_t& hs) const {
	auto order = canonical_xform_order();
	for (auto idx : order) {
		hs.update(0xDULL); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
		xforms_[idx].add_to_hash(hs);
	}

	if (final_xform.has_value()) {
		hs.update(0xFULL); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
		final_xform->add_to_hash(hs);
	}

	if (chaos_table.has_value()) {
		hs.update(0xCULL); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
	}
}

std::vector<std::size_t> rfkt::flame::canonical_xform_order() const {
	auto indicies = std::vector<std::size_t>{};
	indicies.resize(xforms_.size());

	auto hashes = std::vector<rfkt::hash_t>{};
	hashes.reserve(xforms_.size());
	for (const auto& xf : xforms_) {
		hashes.push_back(xf.hash());
	}

	std::iota(indicies.begin(), indicies.end(), 0);
	std::sort(indicies.begin(), indicies.end(), [this, &hashes](std::size_t a, std::size_t b) {
		return hashes[a] < hashes[b];
	});

	return indicies;
}

std::size_t rfkt::flame::size_reals() const noexcept {
	auto size = final_xform ? final_xform->size_reals() : 0;

	if (chaos_table.has_value()) {
		size += xforms_.size() * (xforms_.size() + 1);
	}

	for (const auto& xf : xforms_) {
		size += xf.size_reals();
	}
	return size + 13;
}

std::vector<std::size_t> rfkt::flame::affine_indices() const {

	auto ret = std::vector<std::size_t>{};
	ret.push_back(0);
	ret.push_back(rfkt::affine::size_reals());

	constexpr static std::size_t flame_offset = 13;
	constexpr static std::size_t xform_base_reals = 4;

	std::size_t index = chaos_table.has_value() ? xforms_.size() * (xforms_.size() + 1): 0;
	index += flame_offset;

	auto order = canonical_xform_order();

	for (auto idx : order) {
		auto& xf = xforms_[idx];
		index += xform_base_reals;

		for (auto& vl : xf.vchain) {
			ret.push_back(index);
			index += vl.size_reals();
		}
	}

	if (final_xform) {
		index += xform_base_reals;

		for (auto& vl : final_xform->vchain) {
			ret.push_back(index);
			index += vl.size_reals();
		}
	}

	return ret;
}

rfkt::xform& rfkt::flame::add_xform(xform&& xf) noexcept {

	if (chaos_table.has_value()) {
		for(auto& row: chaos_table.value()) {
			row.emplace_back(1.0);
		}

		chaos_table->emplace_back();
		for(int i = 0; i < xforms_.size(); i++) {
			chaos_table->back().emplace_back(1.0);
		}
	}

	return xforms_.emplace_back(std::move(xf));
}

void rfkt::flame::add_chaos() noexcept {
	if (chaos_table.has_value()) return;

	chaos_table.emplace();
	for(int i = 0; i < xforms_.size(); i++) {
		chaos_table->emplace_back();
		for(int j = 0; j < xforms_.size(); j++) {
			chaos_table->back().emplace_back(1.0);
		}
	}
}

rfkt::anima* rfkt::flame::lookup(std::string_view path) {
	auto [head, tail] = detail::split_path(path);
	if (auto ptr = name_to_pointer(head); ptr) return &(this->*ptr);
	if (head == "xform") {
		auto [xform_idx, xform_path] = detail::split_path(tail);
		if(xform_idx.empty()) return nullptr;
		if(xform_idx == "final") return final_xform ? final_xform->lookup(xform_path) : nullptr;
		int idx = std::stoi(std::string(xform_idx));
		if(idx < 0 || idx >= xforms_.size()) return nullptr;
		return xforms_[idx].lookup(xform_path);
	}
	return nullptr;
}

rfkt::interpolator::interpolator(const rfkt::flame& linit, const rfkt::flame& rinit, const rfkt::flamedb& fdb, bool interp_by_weight)
{
	left.flame = linit;
	right.flame = rinit;

	left.type_hash = linit.hash();
	right.type_hash = rinit.hash();

	left.value_hash = linit.value_hash();
	right.value_hash = rinit.value_hash();

	rebuild_sides(interp_by_weight, fdb);

	for (int i = 0; i < left.flame.palette.size(); i++) {
		auto diff = right.flame.palette[i][0] - left.flame.palette[i][0];

		if (diff > 180) {
			right.flame.palette[i][0] -= 360;
		}
		else if (diff < -180) {
			right.flame.palette[i][0] += 360;
		}
	}
}

void rfkt::interpolator::interp_xforms(rfkt::xform& l, rfkt::xform& r, const rfkt::flamedb& fdb) {

	const auto max_vlinks = std::max(l.vchain.size(), r.vchain.size());

	while (l.vchain.size() < max_vlinks) {
		l.vchain.emplace_back(fdb.make_padder(r.vchain[l.vchain.size()]));
	}

	while (r.vchain.size() < max_vlinks) {
		r.vchain.emplace_back(fdb.make_padder(l.vchain[r.vchain.size()]));
	}

	for (int i = 0; i < max_vlinks; i++) {

		auto& vll = l.vchain[i];
		auto& vlr = r.vchain[i];

		for (const auto& [name, vdata] : vll) {
			if (!vlr.has_variation(name)) {
				vlr.add_variation({ name, vdata });
				vlr[name].weight = 0.0;
			}
		}

		for (const auto& [name, vdata] : vlr) {
			if (!vll.has_variation(name)) {
				vll.add_variation({ name, vdata });
				vll[name].weight = 0.0;
			}
		}
	}
}

void rfkt::interpolator::rebuild_sides(bool interp_by_weight, const rfkt::flamedb& fdb) {
	
	auto nleft = left.flame.xforms().size();
	auto nright = right.flame.xforms().size();

	if (interp_by_weight) {
		for (int i = 0; i < right.flame.xforms().size(); i++) {
			auto xfc = right.flame.xforms()[i];
			left.flame.add_xform(std::move(xfc));
		}

		right.flame.clear_xforms();
		for (int i = 0; i < left.flame.xforms().size(); i++) {
			auto xfc = left.flame.xforms()[i];
			right.flame.add_xform(std::move(xfc));
		}

		for (auto i = nleft; i < left.flame.xforms().size(); i++) {
			left.flame.xforms()[i].weight = 0.0;
		}

		for (auto i = 0; i < nleft; i++) {
			right.flame.xforms()[i].weight = 0.0;
		}
	}
	else {

		auto left_weight_sum = std::accumulate(left.flame.xforms().begin(), left.flame.xforms().end(), 0.0, [](double sum, const auto& xf) { return sum + xf.weight.t0; });
		auto right_weight_sum = std::accumulate(right.flame.xforms().begin(), right.flame.xforms().end(), 0.0, [](double sum, const auto& xf) { return sum + xf.weight.t0; });

		auto xform_distance = [&](const rfkt::xform& l, const rfkt::xform& r) {
			constexpr static auto weight_distance_scale = 0.1;
			constexpr static auto affine_distance_scale = 1.0;
			constexpr static auto affine_distance_decay = 1.5;
			constexpr static auto vlink_similarity_scale = 4.0;
			constexpr static auto vlink_empty_scale = 5.0;


			double dist = weight_distance_scale * std::abs(l.weight.t0/left_weight_sum - r.weight.t0/right_weight_sum);

			if(!l.vchain.empty() && !r.vchain.empty()) {
				dist += affine_distance_scale * (1.0 - std::exp(-l.vchain[0].transform.distance(r.vchain[0].transform) / affine_distance_decay));
				dist += vlink_similarity_scale * (1.0 - l.vchain[0].similarity(&r.vchain[0]));
			} else {
				dist += vlink_empty_scale;
			}

			return dist;
		};

		const auto max_xforms = static_cast<long>(std::max(left.flame.xforms().size(), right.flame.xforms().size()));


		while (left.flame.xforms().size() < max_xforms) {
			left.flame.add_xform({});
		}

		while (right.flame.xforms().size() < max_xforms) {
			right.flame.add_xform({});
		}

		auto cost_matrix = dlib::matrix<std::int64_t>(max_xforms, max_xforms);
		for(int i = 0; i < max_xforms; i++) {
			for(int j = 0; j < max_xforms; j++) {
				constexpr static auto cost_matrix_scale = -1'000'000.0;
				cost_matrix(i, j) = static_cast<std::int64_t>(std::round(xform_distance(left.flame.xforms()[i], right.flame.xforms()[j]) * cost_matrix_scale));
			}
		}

		auto assignment = dlib::max_cost_assignment(cost_matrix);

		std::vector<rfkt::xform> reordered_right(max_xforms);
		for(int i = 0; i < max_xforms; i++) {
			reordered_right[i] = std::move(right.flame.xforms()[assignment[i]]);
			SPDLOG_INFO("xform {} -> {} (cost: {})", i, assignment[i], cost_matrix(i, assignment[i]));
		}

		right.flame.xforms_ = std::move(reordered_right);

		for (int i = 0; i < max_xforms; i++) {
			interp_xforms(left.flame.xforms()[i], right.flame.xforms()[i], fdb);
		}
	}

	if(left.flame.final_xform.has_value() || right.flame.final_xform.has_value()) {
		if (left.flame.final_xform.has_value() && !right.flame.final_xform.has_value()) {
			right.flame.final_xform = rfkt::xform{};
		}
		if (right.flame.final_xform.has_value() && !left.flame.final_xform.has_value()) {
			left.flame.final_xform = rfkt::xform{};
		}
		if (left.flame.final_xform.has_value() && right.flame.final_xform.has_value()) {
			interp_xforms(left.flame.final_xform.value(), right.flame.final_xform.value(), fdb);
		}
	}

	if(left.flame.rotate.t0 - right.flame.rotate.t0 > 180.0) {
		right.flame.rotate.t0 += 360.0;
	}
	else if(left.flame.rotate.t0 - right.flame.rotate.t0 < -180.0) {
		right.flame.rotate.t0 -= 360.0;
	}

	auto left_weight_sum = std::accumulate(left.flame.xforms().begin(), left.flame.xforms().end(), 0.0, [](double sum, const auto& xf) { return sum + xf.weight.t0; });
	auto right_weight_sum = std::accumulate(right.flame.xforms().begin(), right.flame.xforms().end(), 0.0, [](double sum, const auto& xf) { return sum + xf.weight.t0; });

	for(int i = 0; i < left.flame.xforms().size(); i++) {
		left.flame.xforms()[i].weight.t0 /= left_weight_sum;
		right.flame.xforms()[i].weight.t0 /= right_weight_sum;
	}
}

void rfkt::flame_types::bind_to_lua(sol::state& state) {
	using namespace rfkt;

	auto anima_t = state.new_usertype<rfkt::anima>("anima", sol::constructors<rfkt::anima(), rfkt::anima(double)>());
	anima_t["t0"] = &rfkt::anima::t0;

	auto affine_t = state.new_usertype<rfkt::affine>("affine", sol::constructors<rfkt::affine(), rfkt::affine(double, double, double, double, double, double)>());
	affine_t["a"] = &rfkt::affine::a;
	affine_t["b"] = &rfkt::affine::b;
	affine_t["c"] = &rfkt::affine::c;
	affine_t["d"] = &rfkt::affine::d;
	affine_t["e"] = &rfkt::affine::e;
	affine_t["f"] = &rfkt::affine::f;

	auto vardata_t = state.new_usertype<rfkt::vardata>("vardata", sol::no_constructor);
	vardata_t["weight"] = &rfkt::vardata::weight;

	auto vlink_t = state.new_usertype<rfkt::vlink>("vlink", sol::no_constructor);
	vlink_t["transform"] = &rfkt::vlink::transform;
	vlink_t["mod_x"] = &rfkt::vlink::mod_x;
	vlink_t["mod_y"] = &rfkt::vlink::mod_y;
	vlink_t["mod_scale"] = &rfkt::vlink::mod_scale;
	vlink_t["mod_rotate"] = &rfkt::vlink::mod_rotate;

	auto xform_t = state.new_usertype<rfkt::xform>("xform", sol::no_constructor);
	xform_t["weight"] = &rfkt::xform::weight;
	xform_t["color"] = &rfkt::xform::color;
	xform_t["color_speed"] = &rfkt::xform::color_speed;
	xform_t["opacity"] = &rfkt::xform::opacity;
	xform_t["vchain_size"] = [](const rfkt::xform& xf) { return xf.vchain.size(); };
	xform_t["vlink"] = [](rfkt::xform& xf, int idx) -> std::optional<vlink*> { 
		if (idx < 0 || idx >= xf.vchain.size()) return std::nullopt;
		return &xf.vchain[idx]; 
	};

	auto flame_t = state.new_usertype<rfkt::flame>("flame", sol::no_constructor);
	flame_t["center_x"] = &rfkt::flame::center_x;
	flame_t["center_y"] = &rfkt::flame::center_y;
	flame_t["scale"] = &rfkt::flame::scale;
	flame_t["rotate"] = &rfkt::flame::rotate;
	flame_t["gamma"] = &rfkt::flame::gamma;
	flame_t["brightness"] = &rfkt::flame::brightness;
	flame_t["vibrancy"] = &rfkt::flame::vibrancy;

	flame_t["num_xforms"] = [](const rfkt::flame& f) {
		return f.xforms().size();
	};

	flame_t["xform"] = [](rfkt::flame& f, int idx) -> sol::optional<rfkt::xform*> {
		if (idx >= 0 && idx < f.xforms().size())
			return &f.xforms()[idx];
		else
			return sol::nullopt;
	};

	flame_t["final_xform"] = sol::property(
		[](rfkt::flame& f) -> sol::optional<xform*> {
			if(!f.final_xform.has_value()) return sol::nullopt;
			return &f.final_xform.value();
		},
		[](rfkt::flame& f, const rfkt::xform& xf) {
			f.final_xform = xf;
		}
	);

	flame_t["has_final_xform"] = [](const rfkt::flame& f) {
		return f.final_xform.has_value();
	};
}