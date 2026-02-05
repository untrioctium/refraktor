#include <map>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iterator>

#include <spdlog/spdlog.h>

#include "librefrakt/flame_types.hpp"
#include "librefrakt/flame_info.hpp"

#include "flang/grammar.hpp"

nlohmann::json yaml_to_json(const YAML::Node& node) {

	switch(node.Type()) {
		case YAML::NodeType::Null:
			return nullptr;

		case YAML::NodeType::Sequence: {
			auto arr = nlohmann::json::array();
			for(const auto& elem: node) {
				arr.push_back(yaml_to_json(elem));
			}
			return arr;
		}

		case YAML::NodeType::Map: {
			auto obj = nlohmann::json::object();
			for(auto it = node.begin(); it != node.end(); it++) {
				obj[it->first.as<std::string>()] = yaml_to_json(it->second);
			}
			return obj;
		}

		case YAML::NodeType::Scalar: {
			const auto& scalar = node.Scalar();

			try {
				std::size_t pos;
				auto i = std::stoll(scalar, &pos);
				if (pos == scalar.size()) return i;
			} catch(...) {}

			try {
				std::size_t pos;
				auto d = std::stod(scalar, &pos);
				if (pos == scalar.size()) return d;
			} catch(...) {}

			if(scalar == "true" || scalar == "True" || scalar == "TRUE") return true;
			if(scalar == "false" || scalar == "False" || scalar == "FALSE") return false;

			return scalar;
		}

		default:
			throw std::runtime_error("unsupported YAML node type");
	}
}

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

void rfkt::initialize(rfkt::flamedb& fdb, const nlohmann::json& vdefs, const nlohmann::json& cdefs)
{
	auto find_xcommon_calls = [](const flang::ast& src) -> std::set<std::string> {
		using namespace flang::matchers;

		constexpr static auto predicate = of_type<flang::grammar::member_access> and with_child_at<0>(of_type<flang::grammar::variable> and with_content<"common">);

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

	std::vector<std::string> names;
	for (auto it = vdefs.begin(); it != vdefs.end(); it++) {
		names.push_back(it.key());
	}
	std::ranges::sort(names);

	for (auto it = cdefs.begin(); it != cdefs.end(); it++) {
		auto name = it.key();
		fdb.add_or_update_common(name, it.value().get<std::string>());
	}

	for (const auto& name : names) {
		auto def = rfkt::flamedb::variation{};

		def.name = name;

		if (auto node = vdefs[name]; node.is_string()) {
			def.source = node.get<std::string>();
		}
		else {

			def.source = node["src"].get<std::string>();

			if (node.contains("precalc") &&node["precalc"].is_string()) {
				def.precalc_source = node["precalc"].get<std::string>();
			}

			if (node.contains("tags") && node["tags"].is_array()) {
				for (const auto& tag : node["tags"]) {
					auto tag_str = tag.get<std::string>();

					if (tag_str == "pad_affine") {
						def.pad = rfkt::flamedb::pad_type::affine;
					}
					else if (tag_str == "pad_rotated") {
						def.pad = rfkt::flamedb::pad_type::rotated;
					}
					else if (tag_str == "pad_identity") {
						def.pad = rfkt::flamedb::pad_type::identity;
					}
				}
			}

			if (node.contains("param") && node["param"].is_object()) {
				auto param = node["param"];


				for (auto it = param.begin(); it != param.end(); it++) {
					auto pdef = rfkt::flamedb::parameter{};
					pdef.name = it.key();
					auto& value = it.value();

					if (value.contains("default") && value["default"].is_number()) pdef.default_value = value["default"].get<double>();
					if (value.contains("identity") && value["identity"].is_number()) pdef.identity_value = value["identity"].get<double>();

					if (value.contains("tags") && value["tags"].is_array()) {
						for (const auto& tag : value["tags"]) {
							pdef.tags.insert(tag.get<std::string>());
						}
					}

					def.parameters.emplace(pdef.name, std::move(pdef));
				}
			}
		}

		fdb.add_or_update_variation(def);
	}

}

void rfkt::initialize(rfkt::flamedb& fdb, std::string_view config_path)
{
	auto variations_stream = std::ifstream{ std::string{ config_path } + "/variations_new.yml" };
	auto common_stream = std::ifstream{ std::string{ config_path } + "/common.yml" };

	auto variations_data = std::string{ std::istreambuf_iterator<char>{variations_stream}, std::istreambuf_iterator<char>{} };
	auto common_data = std::string{ std::istreambuf_iterator<char>{common_stream}, std::istreambuf_iterator<char>{} };

	auto vdefs = YAML::Load(variations_data);
	auto cdefs = YAML::Load(common_data);

	initialize(fdb, yaml_to_json(vdefs), yaml_to_json(cdefs));
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

		if (auto it = variations_.find(vdef.name); it != variations_.end() && it->second.pad) {
			pad_tags_[it->second.pad.value()].erase(it->second.name);
		}

		if (vdef.pad) {
			pad_tags_[vdef.pad.value()].insert(vdef.name);
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
		std::map<std::string, anima, std::less<>> params;
		for (const auto& [name, param] : vdef.parameters) {
			if (param.tags.contains("precalc")) {
				precalc_count++;
				continue;
			}
			params[name] = param.default_value.value_or(0.0);
		}
		return { vdef.name, { 0.0, precalc_count, std::move(params)}};
	}

	auto flamedb::make_padder(const vlink& vl) const -> vlink
	{
		auto ret = vl;
		ret.transform = affine::identity();
		ret.mod_rotate = { 0.0 };
		ret.mod_scale = { 1.0 };
		ret.mod_x = { 0.0 };
		ret.mod_y = { 0.0 };


		for (auto& [name, vd] : ret) {
			vd.weight = 0.0;
		}

		for (auto& vname : pad_tags_.at(pad_type::rotated)) {
			if(!vl.has_variation(vname)) continue;

			if (!ret.has_variation("linear")) {
				ret.add_variation(make_vardata("linear"));
			}

			ret["linear"].weight = -1.0;
			ret.transform.a = -1.0;
			ret.transform.e = -1.0;
			
			return ret;
		}

		int found = 0;

		for (auto& vname : pad_tags_.at(pad_type::identity)) {
			if (!vl.has_variation(vname)) continue;

			ret[vname].weight = 1.0;
			found++;

			for (auto& [pname, param] : get_variation(vname).parameters) {
				if (param.identity_value.has_value()) {
					ret[vname][pname] = param.identity_value.value();
				}
			}
		}

		if (found > 0) {
			for (auto& [name, vd] : ret) {
				vd.weight.t0 /= found;
			}

			return ret;
		}

		for (auto& vname : pad_tags_.at(pad_type::affine)) {
			if (!vl.has_variation(vname)) continue;

			ret[vname].weight = 1.0;
			found++;
		}

		if (found > 0) {

			ret.transform.a = 0.0;
			ret.transform.d = 1.0;
			ret.transform.b = 1.0;
			ret.transform.e = 0.0;
			ret.transform.c = 0.0;
			ret.transform.f = 0.0;

			for (auto& [name, vd] : ret) {
				vd.weight.t0 /= found;
			}

			return ret;
		}

		if (!ret.has_variation("linear")) {
			ret.add_variation(make_vardata("linear"));
		}

		ret["linear"].weight = 1.0;

		return ret;
	}

	std::string flamedb::serialize() const noexcept {
		nlohmann::json js;

		auto variations_js = nlohmann::json::object();
		for (const auto& [vname, var] : variations_) {
			auto var_js = nlohmann::json::object();
			var_js["src"] = var.source;
			if (var.precalc_source) {
				var_js["precalc"] = var.precalc_source.value();
			}

			if(var.parameters.size() > 0) {
				auto parameters_js = nlohmann::json::object();

				for (const auto& [pname, pdef] : var.parameters) {
					auto pdef_js = nlohmann::json::object();
					if (pdef.default_value) {
						pdef_js["default"] = pdef.default_value.value();
					}
					if (pdef.identity_value) {
						pdef_js["identity"] = pdef.identity_value.value();
					}
					if (pdef.tags.size() > 0) {
						pdef_js["tags"] = nlohmann::json::array();
						for (const auto& tag : pdef.tags) {
							pdef_js["tags"].push_back(tag);
						}
					}
					parameters_js[pname] = std::move(pdef_js);
				}
				var_js["parameters"] = std::move(parameters_js);
			}

			if(var.tags.size() > 0) {
				var_js["tags"] = nlohmann::json::array();
				for (const auto& tag : var.tags) {
					var_js["tags"].push_back(tag);
				}
			}

			variations_js[vname] = std::move(var_js);

		}

		js["variations"] = std::move(variations_js);


		auto common_js = nlohmann::json::object();
		for (const auto& [cname, cdef] : common_) {
			common_js[cname] = cdef;
		}
		js["common"] = std::move(common_js);

		return js.dump(4);
	}
}