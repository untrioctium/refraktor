#include <map>
#include <yaml-cpp/yaml.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/util/string.h>
#include <librefrakt/animators.h>

struct flame_table {
	std::vector<rfkt::flame_info::def::variation> variations;
	std::vector<rfkt::flame_info::def::parameter> parameters;
	std::map<std::string, rfkt::flame_info::def::common, std::less<>> common;

	std::map<std::string, std::reference_wrapper<rfkt::flame_info::def::variation>, std::less<>> variation_names;
	std::map<std::string, std::reference_wrapper<rfkt::flame_info::def::parameter>, std::less<>> parameter_names;
};

auto& ft() {
	static auto instance = flame_table{};
	return instance;
}

void rfkt::flame_info::initialize(std::string_view config_path)
{
	rfkt::animator::init_builtins();

	auto find_xcommon_calls = [](const std::string& src) {
		return str_util::find_unique(src, R"regex(xcommon\(\s*([a-zA-z0-9_]+)\s*\))regex");
	};

	auto defs = YAML::LoadFile(std::string{ config_path });

	std::size_t num_variations = 0;
	std::size_t num_parameters = 0;

	// FIXME: this clears up issues with the vectors moving their data and destroying references when resizing
	// but this should be properly calculated instead of allocating way more space than needed
	ft().variations.reserve(256);
	ft().parameters.reserve(256);

	auto vdefs = defs["variations"];
	std::vector<std::string> names;
	for (auto it = vdefs.begin(); it != vdefs.end(); it++) {
		names.push_back(it->first.as<std::string>());
	}
	std::sort(std::begin(names), std::end(names));

	auto common = defs["xcommon"];
	for (auto it = common.begin(); it != common.end(); it++) {
		auto name = it->first.as<std::string>();
		auto src = it->second.as<std::string>();

		auto cdef = flame_info::def::common{};
		cdef.name = name;
		cdef.source = src;
		cdef.dependencies = find_xcommon_calls(src);
		ft().common[name] = cdef;
	}

	for (auto& name : names) {
		auto def = def::variation{};
		def.index = (num_variations++);
		def.name = name;

		auto yml = vdefs[name];

		if (yml.IsScalar()) {
			def.source = yml.as<std::string>();
		}
		else {
			auto node = yml.as<YAML::Node>();
			def.source = node["src"].as<std::string>();

			auto param = node["param"];

			for (auto it = param.begin(); it != param.end(); it++) {
				auto pdef = def::parameter{};
				pdef.name = it->first.as<std::string>();;

				if (it->second["default"].IsScalar()) pdef.default_value = it->second["default"].as<double>();

				pdef.index = (num_parameters++);
				ft().parameters.push_back(pdef);
				ft().parameter_names.insert_or_assign(pdef.name, std::ref(ft().parameters[pdef.index]));
				def.parameters.push_back(std::ref(ft().parameters.at(pdef.index)));
			}

			auto flags = node["flags"];
			for (auto it = flags.begin(); it != flags.end(); it++)
				def.flags.insert(it->as<std::string>());
		}
		for (const auto& name : find_xcommon_calls(def.source)) {
			def.dependencies.push_back(ft().common[name]);
		}

		ft().variations.push_back(def);
		ft().variation_names.insert_or_assign(def.name, std::ref(ft().variations.at(def.index)));
	}

}

auto rfkt::flame_info::num_variations() -> std::size_t
{
	return ft().variations.size();
}

auto rfkt::flame_info::num_parameters() -> std::size_t
{
	return ft().parameters.size();
}

bool rfkt::flame_info::is_variation(std::string_view name)
{
	return ft().variation_names.contains(name);
}

bool rfkt::flame_info::is_parameter(std::string_view name)
{
	return ft().parameter_names.contains(name);
}

bool rfkt::flame_info::is_common(std::string_view name)
{
	return ft().common.contains(name);
}

auto rfkt::flame_info::variation(std::size_t idx) -> const def::variation&
{
	return ft().variations.at(idx);
}

auto rfkt::flame_info::variation(std::string_view name) -> const def::variation&
{
	return ft().variation_names.find(name)->second;
}

auto rfkt::flame_info::parameter(std::size_t idx) -> const def::parameter&
{
	return ft().parameters.at(idx);
}

auto rfkt::flame_info::parameter(std::string_view name) -> const def::parameter&
{
	return ft().parameter_names.find(name)->second;
}

auto rfkt::flame_info::common(std::string_view name) -> const def::common&
{
	return ft().common.find(name)->second;
}

auto rfkt::flame_info::variations() -> const std::vector<def::variation>&
{
	return ft().variations;
}
