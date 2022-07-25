#include <pugixml.hpp>

#include <librefrakt/util/string.h>
#include <librefrakt/util/color.h>
#include <librefrakt/flame_types.h>

rfkt::flame::flame(const flame& o)
{
	(*this) = o;
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