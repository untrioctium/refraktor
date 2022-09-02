#include <pugixml.hpp>

#include <librefrakt/util/string.h>
#include <librefrakt/util/color.h>
#include <librefrakt/flame_types.h>

constexpr auto split_path(std::string_view sv) -> std::pair<std::string_view, std::string_view> {

	auto pos = sv.find('/');
	if (pos == std::string_view::npos) return { sv, {} };

	auto head = sv.substr(0, pos);
	if (pos + 1 >= sv.length()) return { head, {} };

	return { head, sv.substr(pos + 1) };
}

constexpr int to_int(std::string_view sv) {
	int ret = 0;
	for (const auto ch : sv) {
		if (ch < '0' || ch > '9') return -1;
		ret *= 10;
		ret += ch - '0';
	}

	return ret;
}

rfkt::flame::flame(const flame& o)
{
	(*this) = o;
}

rfkt::animated_double* rfkt::flame::seek(std::string_view path)
{
	auto [head, tail] = split_path(path);

	switch (head[0]) {
	case 'x': return (head.size() == 1 && tail.empty()) ? &center.first : nullptr;
	case 'y': return (head.size() == 1 && tail.empty()) ? &center.second : nullptr;
	case 's': return (head.size() == 1 && tail.empty()) ? &scale : nullptr;
	case 'r': return (head.size() == 1 && tail.empty()) ? &rotate : nullptr;
	case 'g': return (head.size() == 1 && tail.empty()) ? &gamma : nullptr;
	case 'v': return (head.size() == 1 && tail.empty()) ? &vibrancy : nullptr;
	case 'b': return (head.size() == 1 && tail.empty()) ? &brightness : nullptr;
	case 'f': return (head.size() == 1 && !tail.empty() && final_xform.has_value()) ? final_xform->seek(tail) : nullptr;
	default: break;
	}

	if (auto idx = to_int(head); idx >= 0 && !tail.empty() && idx < xforms.size()) {
		return xforms[idx].seek(tail);
	}

	return nullptr;
}

std::string rfkt::flame::dump() const
{
	auto ret = std::string{};
	
	ret += fmt::format("x {}\n", center.first.t0);
	ret += fmt::format("y {}\n", center.second.t0);
	ret += fmt::format("s {}\n", scale.t0);
	ret += fmt::format("r {}\n", rotate.t0);
	ret += fmt::format("g {}\n", gamma.t0);
	ret += fmt::format("v {}\n", vibrancy.t0);
	ret += fmt::format("b {}\n", brightness.t0);

	for (int i = 0; i < xforms.size(); i++) {
		auto prefix = fmt::format("{}", i);
		ret += xforms[i].dump(prefix);
	}

	if (final_xform) ret += final_xform->dump("f");

	return ret;
}

std::unique_ptr<rfkt::animator> make_xform_rotator() {
	static const json args = []() {
		auto o = json::object();
		o.emplace("per_loop", 360.0);
		return o;
	}();

	return rfkt::animator::make("increase", args);
}


auto rfkt::flame::import_flam3(const std::string& path) -> std::optional<flame>
{
	pugi::xml_document fxml;
	auto result = fxml.load_file(path.c_str());

	auto f = std::optional<flame>{ std::nullopt };
	if (!result) {
		return std::nullopt;
	}

	f = flame{};

	auto fnode = fxml.child("flame");

	auto center_str = str_util::split(fnode.attribute("center").as_string());
	auto size_str = str_util::split(fnode.attribute("size").as_string());

	auto scale_div = std::stod(size_str[1]);

	f->center.first = std::stod(center_str[0]);
	f->center.second = std::stod(center_str[1]);
	f->scale = fnode.attribute("scale").as_double() / scale_div;
	f->rotate = fnode.attribute("rotate").as_double();
	f->brightness = fnode.attribute("brightness").as_double();
	f->gamma = fnode.attribute("gamma").as_double();
	f->vibrancy = fnode.attribute("vibrancy").as_double();

	bool is_bad = false;

	for (auto& node : fnode.children()) {
		std::string nname = node.name();

		if (nname == "xform" || nname == "finalxform") {
			auto xf = xform{};

			auto vlinks = std::array<vlink, 3>{};
			bool has_post_affine = false;

			for (auto& attr : node.attributes()) {
				std::string aname = attr.name();
				char which_vl = 1;

				if (aname.starts_with("pre_")) {
					which_vl = 0;
					aname = aname.substr(4);
				}
				else if (aname.starts_with("post_")) {
					which_vl = 2;
					aname = aname.substr(5);
				}

				if (aname == "weight") xf.weight = attr.as_double();
				else if (aname == "color") xf.color = attr.as_double();
				else if (aname == "color_speed") xf.color_speed = attr.as_double();
				else if (aname == "animate") {
					if(attr.as_double() > 0.0)
						vlinks[1].aff_mod_rotate.ani = make_xform_rotator(); 
				}
				else if (aname == "opacity") xf.opacity = attr.as_double();
				else if (aname == "coefs") vlinks[1].affine = affine_matrix::from_strings(str_util::split(attr.as_string()));
				else if (aname == "post") {
					vlinks[2].affine = affine_matrix::from_strings(str_util::split(attr.as_string()));
					has_post_affine = true;
				}
				else if (flame_info::is_parameter(aname)) {
					vlinks[which_vl].parameters[flame_info::parameter(aname).index] = attr.as_double();
				}
				else if (flame_info::is_variation(aname)) {
					vlinks[which_vl].variations[flame_info::variation(aname).index] = attr.as_double();
				}
				else if (aname != "chaos") { SPDLOG_ERROR("Unknown attribute {} in flame {}", aname, path); is_bad = true; }
			}

			if (vlinks[0].variations.size() > 0) {
				xf.vchain.push_back(vlinks[0]);
			}
			xf.vchain.push_back(vlinks[1]);
			if (vlinks[2].variations.size() > 0 || has_post_affine) {
				if (vlinks[2].variations.size() == 0) vlinks[2].variations[flame_info::variation("linear").index] = 1.0;
				xf.vchain.push_back(vlinks[2]);
			}

			/*for (auto motion : node.children()) {
				for (auto attr : motion.attributes()) {
					std::string aname = attr.name();
					if (flame::is_variation(aname)) {
						auto vdef = flame::variation(aname);
						if (!xf.has_variation(vdef.index)) xf.variations[vdef.index] = 0.0;
					}
				}
			}*/

			if (nname == "xform") f->xforms.push_back(xf);
			else f->final_xform = xf;
		}
		else if (nname == "color") {
			auto idx = node.attribute("index").as_ullong();
			auto split_rgb = str_util::split(node.attribute("rgb").as_string());
			auto rgb = uchar3{
				(std::uint8_t)(round(std::stod(split_rgb[0]))),
				(std::uint8_t)(round(std::stod(split_rgb[1]))),
				(std::uint8_t)(round(std::stod(split_rgb[2])))
			};

			auto hsv = color::rgb_to_hsv(rgb);
			f->palette()[idx][0] = hsv.x;
			f->palette()[idx][1] = hsv.y;
			f->palette()[idx][2] = hsv.z;
		}
	}
	if (is_bad) return std::nullopt;

	return f;
}

auto rfkt::flame::operator=(const flame& o) noexcept -> flame& {
	center = o.center;
	scale = o.scale;
	rotate = o.rotate;

	xforms = o.xforms;
	final_xform = o.final_xform;

	gamma = o.gamma;
	vibrancy = o.vibrancy;
	brightness = o.brightness;

	palette() = o.palette();
	return *this;
}

rfkt::animated_double* rfkt::vlink::seek(std::string_view path)
{
	auto [head, tail] = split_path(path);

	if (tail.empty() || head.size() > 1) return nullptr;

	switch (head[0]) {
	case 'v': {
		if (!rfkt::flame_info::is_variation(tail)) return nullptr;

		auto iter = variations.find(rfkt::flame_info::variation(tail).index);
		if (iter == variations.end()) return nullptr;
		return &iter->second;
	}
	case 'p': {
		if (!rfkt::flame_info::is_parameter(tail)) return nullptr;

		auto iter = parameters.find(rfkt::flame_info::parameter(tail).index);
		if (iter == parameters.end()) return nullptr;
		return &iter->second;
	}
	case 'a':
		if (tail.size() > 1) return nullptr;
		switch (tail[0]) {
		case 'a': return &affine.a;
		case 'b': return &affine.b;
		case 'c': return &affine.c;
		case 'd': return &affine.d;
		case 'e': return &affine.e;
		case 'f': return &affine.f;
		default: return nullptr;
		}

	case 'm':
		if (tail.size() > 1) return nullptr;
		switch (tail[0]) {
		case 'r': return &aff_mod_rotate;
		case 's': return &aff_mod_scale;
		case 'x': return &aff_mod_translate.first;
		case 'y': return &aff_mod_translate.second;
		default: return nullptr;
		}

	default: return nullptr;
	}
}

std::string rfkt::vlink::dump(std::string_view prefix) const
{
	auto ret = std::string{};
	auto out = [&ret, prefix](auto name, const auto& value) {
		ret += fmt::format("{}/{} {}\n", prefix, name, value.t0);
	};

	out("a/a", affine.a);
	out("a/b", affine.b);
	out("a/c", affine.c);
	out("a/d", affine.d);
	out("a/e", affine.e);
	out("a/f", affine.f);

	out("m/r", aff_mod_rotate);
	out("m/s", aff_mod_scale);
	out("m/x", aff_mod_translate.first);
	out("m/y", aff_mod_translate.second);

	for (auto& [k, v] : variations) {
		out(fmt::format("v/{}", rfkt::flame_info::variation(k).name), v);
	}

	for (auto& [k, v] : parameters) {
		out(fmt::format("p/{}", rfkt::flame_info::parameter(k).name), v);
	}

	return ret;
}

rfkt::animated_double* rfkt::xform::seek(std::string_view path)
{
	auto [head, tail] = split_path(path);

	switch (head[0]) {
	case 'w': return (head.size() == 1 && tail.empty()) ? &weight : nullptr;
	case 'c':
		if (head.size() == 1) return &color;
		if (head.size() != 2) return nullptr;

		if (head[1] == 's') return &color_speed;

		return nullptr;

	case 'o': return (head.size() == 1 && tail.empty()) ? &opacity : nullptr;
	default: break;
	}

	if (auto idx = to_int(head); idx >= 0 && !tail.empty() && idx < vchain.size()) {
		return vchain[idx].seek(tail);
	}

	return nullptr;
}

std::string rfkt::xform::dump(std::string_view prefix) const
{
	auto ret = std::string{};
	auto out = [&ret, prefix](auto name, const auto& value) {
		ret += fmt::format("{}/{} {}\n", prefix, name, value.t0);
	};

	out("w", weight);
	out("c", color);
	out("cs", color_speed);
	out("o", opacity);

	for (int i = 0; i < vchain.size(); i++) {
		ret += vchain[i].dump(fmt::format("{}/{}", prefix, i));
	}

	return ret;
}
