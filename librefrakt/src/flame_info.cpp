#include <map>
#include <yaml-cpp/yaml.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/util/string.h>
#include <librefrakt/animators.h>

#include "flang/grammar.h"

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

auto validate_variation(const rfkt::flame_info::def::variation& v) {

	auto scopes = flang::scope_stack{};
	auto& scope = scopes.emplace_back();

	scope.add_vec2("p");
	scope.add_vec2("result");
	scope.add_decimal("weight");

	auto common_group = flang::type_desc::group{};
	for (const auto& [name, def] : ft().common) {
		common_group.add_decimal(name);
	}

	scope.members.emplace("common", std::move(common_group));

	auto param_group = flang::type_desc::group{};
	for (const auto& p : v.parameters) {
		const auto& name = p.get().name;
		auto trimmed = name.substr(name.find(v.name) + v.name.size() + 1);
		param_group.add_decimal(trimmed);
	}

	scope.members.emplace("param", std::move(param_group));

	auto affine_group = flang::type_desc::group{};
	affine_group.add_decimal("a");
	affine_group.add_decimal("b");
	affine_group.add_decimal("c");
	affine_group.add_decimal("d");
	affine_group.add_decimal("e");
	affine_group.add_decimal("f");
	scope.members.emplace("aff", std::move(affine_group));

	auto src_result = flang::validate(v.source.head(), flang::stdlib(), scopes);

	if (src_result.has_value()) {
		SPDLOG_ERROR("Failed to validate variation `{}` source: {}", v.name, src_result->what());
		exit(1);
	}

	if (v.precalc_source.has_value()) {
		auto precalc_result = flang::validate(v.precalc_source->head(), flang::stdlib(), scopes);

		if (precalc_result.has_value()) {
			SPDLOG_ERROR("Failed to validate variation `{}` precalc: {}", v.name, precalc_result->what());
			exit(1);
		}
	}
}

void rfkt::flame_info::initialize(std::string_view config_path)
{
	rfkt::animator::init_builtins();

	auto find_xcommon_calls = [](const flang::ast& src) -> std::set<std::string> {
		using namespace flang::matchers;

		auto predicate = of_type<flang::grammar::member_access>() && with_child(of_type<flang::grammar::variable>() && with_content("common") && of_rank(0));

		std::set<std::string> deps;
		for (const auto& node : src) {
			if (predicate(&node)) {
				const auto& name = node.nth(1)->content();
				if(!deps.contains(name))
					deps.emplace(node.nth(1)->content());
			}
		}
		return deps;
	};

	auto vdefs = YAML::LoadFile(std::string{ config_path } + "/variations_new.yml");
	auto cdefs = YAML::LoadFile(std::string{ config_path } + "/common.yml");

	std::size_t num_variations = 0;
	std::size_t num_parameters = 0;

	// FIXME: this clears up issues with the vectors moving their data and destroying references when resizing
	// but this should be properly calculated instead of allocating way more space than needed
	ft().variations.reserve(256);
	ft().parameters.reserve(256);

	std::vector<std::string> names;
	for (auto it = vdefs.begin(); it != vdefs.end(); it++) {
		names.push_back(it->first.as<std::string>());
	}
	std::sort(std::begin(names), std::end(names));

	for (auto it = cdefs.begin(); it != cdefs.end(); it++) {
		auto name = it->first.as<std::string>();
		auto parsed_src = flang::ast::parse_expression(it->second.as<std::string>());
		if (!parsed_src) {
			throw parsed_src.error();
		}

		auto cdef = flame_info::def::common{};
		cdef.name = name;
		cdef.source = std::move(parsed_src.value());
		cdef.dependencies = find_xcommon_calls(cdef.source);
		ft().common[name] = std::move(cdef);
	}

	for (auto& name : names) {
		auto def = def::variation{};
		def.index = (num_variations++);
		def.name = name;

		auto yml = vdefs[name];

		if (yml.IsScalar()) {
			auto parsed = flang::ast::parse_statement(yml.as<std::string>());
			if (!parsed) {
				throw parsed.error();
			}
			def.source = std::move(parsed.value());
		}
		else {
			auto node = yml.as<YAML::Node>();
			auto parsed_src = flang::ast::parse_statement(node["src"].as<std::string>());
			if (!parsed_src) throw parsed_src.error();
			def.source = std::move(parsed_src.value());

			if (node["precalc"].IsScalar()) {
				auto parsed_precalc = flang::ast::parse_statement(node["precalc"].as<std::string>());
				if(!parsed_precalc) throw std::runtime_error{ "Failed to parse variation: " + name };
				def.precalc_source = std::move(parsed_precalc.value());
			}

			auto param = node["param"];

			for (auto it = param.begin(); it != param.end(); it++) {
				auto pdef = def::parameter{};
				pdef.name = std::format("{}_{}", name, it->first.as<std::string>());

				if (it->second["default"].IsScalar()) pdef.default_value = it->second["default"].as<double>();
				if (it->second["tags"].IsSequence()) {
					for (const auto& tag : it->second["tags"]) {
						if (tag.as<std::string>() == "precalc") {
							pdef.is_precalc = true;
						}
					}
				}

				pdef.index = (num_parameters++);
				pdef.owner = def.index;

				ft().parameters.push_back(pdef);
				ft().parameter_names.insert_or_assign(pdef.name, std::ref(ft().parameters[pdef.index]));
				def.parameters.push_back(std::ref(ft().parameters.at(pdef.index)));
			}

			auto flags = node["flags"];
			for (auto it = flags.begin(); it != flags.end(); it++)
				def.flags.insert(it->as<std::string>());
		}
		for (const auto& xname : find_xcommon_calls(def.source)) {
			def.dependencies.push_back(ft().common[xname]);
		}

		validate_variation(def);

		auto index = def.index;
		ft().variations.emplace_back(std::move(def));
		ft().variation_names.insert_or_assign(name, std::ref(ft().variations.at(index)));
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

auto rfkt::flame_info::variation(std::uint32_t idx) -> const def::variation&
{
	return ft().variations.at(idx);
}

auto rfkt::flame_info::variation(std::string_view name) -> const def::variation&
{
	return ft().variation_names.find(name)->second;
}

auto rfkt::flame_info::parameter(std::uint32_t idx) -> const def::parameter&
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