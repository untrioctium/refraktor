#include <map>
#include <yaml-cpp/yaml.h>

#include <spdlog/spdlog.h>

#include "librefrakt/flame_types.h"
#include "librefrakt/flame_info.h"

#include "flang/grammar.h"

std::optional<flang::semantic_error> validate_variation(const rfkt::flamedb::variation& v, std::span<std::string_view> common, const std::pair<flang::ast, std::optional<flang::ast>>& ast) {

	auto scopes = flang::scope_stack{};
	auto& scope = scopes.emplace_back();

	scope.add_vec2("p");
	scope.add_vec2("result");
	scope.add_decimal("weight");

	auto common_group = flang::type_desc::group{};
	for (auto name : common) {
		common_group.add_decimal(name);
	}

	scope.members.emplace("common", std::move(common_group));

	auto param_group = flang::type_desc::group{};
	for (const auto& [name, def] : v.parameters) {
		param_group.add_decimal(name);
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

	auto src_result = flang::validate(ast.first.head(), flang::stdlib(), scopes);

	if (src_result.has_value()) {
		return src_result;
	}

	if (v.precalc_source.has_value()) {
		auto precalc_result = flang::validate(ast.second->head(), flang::stdlib(), scopes);

		if (precalc_result.has_value()) {
			return precalc_result;
		}
	}

	return std::nullopt;
}

void rfkt::initialize(rfkt::flamedb& fdb, std::string_view config_path)
{
	auto find_xcommon_calls = [](const flang::ast& src) -> std::set<std::string> {
		using namespace flang::matchers;

		constexpr static auto predicate = of_type<flang::grammar::member_access> and with_child(of_type<flang::grammar::variable> and with_content<"common"> and of_rank<0>);

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

	std::vector<std::string> names;
	for (auto it = vdefs.begin(); it != vdefs.end(); it++) {
		names.push_back(it->first.as<std::string>());
	}
	std::sort(std::begin(names), std::end(names));

	for (auto it = cdefs.begin(); it != cdefs.end(); it++) {
		auto name = it->first.as<std::string>();
		fdb.add_or_update_common(name, it->second.as<std::string>());
	}

	for (auto& name : names) {
		auto def = rfkt::flamedb::variation{};

		def.name = name;

		auto yml = vdefs[name];

		if (yml.IsScalar()) {
			def.source = std::move(yml.as<std::string>());
		}
		else {
			auto node = yml.as<YAML::Node>();

			def.source = node["src"].as<std::string>();

			if (node["precalc"].IsScalar()) {
				def.precalc_source = node["precalc"].as<std::string>();
			}

			auto param = node["param"];

			for (auto it = param.begin(); it != param.end(); it++) {
				auto pdef = rfkt::flamedb::parameter{};
				pdef.name = it->first.as<std::string>();

				if (it->second["default"].IsScalar()) pdef.default_value = it->second["default"].as<double>();
				if (it->second["tags"].IsSequence()) {
					for (const auto& tag : it->second["tags"]) {
						pdef.tags.insert(tag.as<std::string>());
					}
				}

				def.parameters.emplace(pdef.name, std::move(pdef));
			}
		}

		fdb.add_or_update_variation(def);
	}

}

namespace rfkt {

	bool flamedb::add_or_update_variation(const variation& vdef) noexcept {
		cache_value_type vsrc;

		auto src = flang::ast::parse_statement(vdef.source);
		if (!src) {
			SPDLOG_ERROR("cannot parse variation `{}`: {}", vdef.name, src.error().what());
			return false;
		}
		vsrc.first = std::move(src.value());

		if (vdef.precalc_source) {
			auto precalc = flang::ast::parse_statement(*vdef.precalc_source);
			if (!precalc) {
				SPDLOG_ERROR("cannot parse variation `{}` precalc: {}", vdef.name, precalc.error().what());
				return false;
			}
			vsrc.second = std::move(precalc.value());
		}

		std::vector<std::string_view> common_names;
		for (const auto& [name, _] : common()) {
			common_names.push_back(name);
		}

		auto validate_result = validate_variation(vdef, common_names, vsrc);
		if (validate_result.has_value()) {
			SPDLOG_ERROR("cannot validate variation `{}`: {}", vdef.name, validate_result.value().what());
			return false;
		}

		variations_[vdef.name] = vdef;
		variation_cache_.insert_or_assign(vdef.name, std::move(vsrc));
		recalc_hash();
		return true;
	}

	bool flamedb::add_or_update_common(std::string_view name, std::string_view source) noexcept {
		auto ast = flang::ast::parse_expression(source);
		if (!ast) {
			SPDLOG_ERROR("cannot parse common `{}`: {}", name, ast.error().what());
			return false;
		}

		// TODO: validate common
		common_.insert_or_assign(std::string{ name }, std::string{ source });
		common_cache_.insert_or_assign(std::string{ name}, std::move(ast.value()));
		recalc_hash();
		return true;
	}

	void flamedb::recalc_hash() noexcept {
		rfkt::hash::state_t hs;
		for (const auto& [_, var] : variations_) {
			var.add_to_hash(hs);
		}

		for (const auto& [name, src] : common_) {
			hs.update(name);
			hs.update(src);
		}

		hash_ = hs.digest();
	}

	auto flamedb::make_vardata(std::string_view vname) const noexcept -> std::pair<std::string, vardata> {
		const auto& vdef = variations_.find(vname)->second;

		auto precalc_count = std::size_t{ 0 };
		std::map<std::string, double, std::less<>> params;
		for (const auto& [name, param] : vdef.parameters) {
			if (param.tags.contains("precalc")) {
				precalc_count++;
				continue;
			}
			params[name] = param.default_value;
		}
		return { vdef.name, { 0.0, precalc_count, std::move(params)}};
	}
}