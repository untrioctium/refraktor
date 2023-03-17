#include "flang/grammar.h"

const flang::detail::operator_map& flang::detail::op_type_map() {

	static const operator_map opm = []() {

		operator_map ops{};

		using enum vtype;

		ops[grammar::demangle<grammar::op::plus>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { vec2, vec2 }, vec2 },
			{ { vec3, vec3 }, vec3 },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, decimal }
		};

		ops[grammar::demangle<grammar::op::minus>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { vec2, vec2 }, vec2 },
			{ { vec3, vec3 }, vec3 },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, decimal }
		};

		ops[grammar::demangle<grammar::op::times>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, decimal },
			{ { integer, vec2 }, vec2 },
			{ { vec2, integer }, vec2 },
			{ { decimal, vec2 }, vec2 },
			{ { vec2, decimal }, vec2 },
			{ { integer, vec3 }, vec3 },
			{ { vec3, integer }, vec3 },
			{ { decimal, vec3 }, vec3 },
			{ { vec3, decimal }, vec3 },
		};

		ops[grammar::demangle<grammar::op::divided>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, decimal },
			{ { vec2, integer }, vec2 },
			{ { vec2, decimal }, vec2 },
			{ { vec3, integer }, vec3 },
			{ { vec3, decimal }, vec3 }
		};

		ops[grammar::demangle<grammar::op::exponent>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { integer, decimal }, decimal },
			{ { decimal, integer }, decimal }
		};

		ops[grammar::demangle<grammar::op::equals>()] = {
			{ { integer, integer }, boolean },
			{ { decimal, decimal }, boolean },
			{ { integer, decimal }, boolean },
			{ { decimal, integer }, boolean },
			{ { boolean, boolean }, boolean }
		};

		ops[grammar::demangle<grammar::op::not_equals>()] = ops[grammar::demangle<grammar::op::equals>()];

		auto comparison_base = std::map<std::pair<vtype, vtype>, vtype>{
			{ { integer, integer }, boolean },
			{ { decimal, decimal }, boolean },
			{ { decimal, integer }, boolean },
			{ { integer, decimal }, boolean }
		};

		ops[grammar::demangle<grammar::op::less_than>()] = comparison_base;
		ops[grammar::demangle<grammar::op::less_equal>()] = comparison_base;
		ops[grammar::demangle<grammar::op::greater_than>()] = comparison_base;
		ops[grammar::demangle<grammar::op::greater_equal>()] = comparison_base;

		ops[grammar::demangle<grammar::op::b_and>()] = { { { boolean, boolean }, boolean } };
		ops[grammar::demangle<grammar::op::b_or>()] = { { { boolean, boolean }, boolean } };

		ops[grammar::demangle<grammar::op::assignment>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { vec2, vec2 }, vec2 },
			{ { vec3, vec3 }, vec3 },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, integer }
		};

		ops[grammar::demangle<grammar::op::plus_assign>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { vec2, vec2 }, vec2 },
			{ { vec3, vec3 }, vec3 },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, integer }
		};

		ops[grammar::demangle<grammar::op::minus_assign>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { vec2, vec2 }, vec2 },
			{ { vec3, vec3 }, vec3 },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, integer }
		};

		ops[grammar::demangle<grammar::op::times_assign>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, integer },
			{ { vec2, integer }, vec2 },
			{ { vec2, decimal }, vec2 },
			{ { vec3, integer }, vec3 },
			{ { vec3, decimal }, vec3 },
		};

		ops[grammar::demangle<grammar::op::divided_assign>()] = {
			{ { integer, integer }, integer },
			{ { decimal, decimal }, decimal },
			{ { decimal, integer }, decimal },
			{ { integer, decimal }, integer },
			{ { vec2, integer }, vec2 },
			{ { vec2, decimal }, vec2 },
			{ { vec3, integer }, vec3 },
			{ { vec3, decimal }, vec3 }
		};

		return ops;

	}();

	return opm;

}

const std::map<std::string_view, flang::vtype>& flang::detail::basic_type_map() {
	const static auto tm = std::map<std::string_view, vtype>{
		{ grammar::demangle<grammar::lit::integer>(), vtype::integer },
		{ grammar::demangle<grammar::lit::decimal>(), vtype::decimal },
		{ grammar::demangle<grammar::lit::boolean>(), vtype::boolean }
	};

	return tm;
}

std::expected<const flang::type_desc::info*, flang::semantic_error> flang::resolve(const ast_node* node, const var_definitions& globals, const scope_stack& scopes) {

	if (node->is_type<grammar::variable>()) {
		if (auto type = globals.members.find(node->content()); type != globals.members.end()) {
			return &type->second;
		}

		for (const auto& scope : scopes) {
			if (auto type = scope.members.find(node->content()); type != scope.members.end()) {
				return &type->second;
			}
		}

		return semantic_error::make<"undeclared variable `{}`">(node, node->content());
	}

	if (node->is_type<grammar::member_access>()) {
		const auto& var = node->nth(0)->content();
		const auto& member = node->nth(1)->content();

		const type_desc::info* desc = [&]() -> const type_desc::info* {
			if (auto type = globals.members.find(var); type != globals.members.end()) {
				return &type->second;
			}

			for (const auto& scope : scopes) {
				if (auto type = scope.members.find(var); type != scope.members.end()) {
					return &type->second;
				}
			}

			return nullptr;
		}();

		if (!desc) {
			return semantic_error::make<"undeclared variable `{}`">(node, var);;
		}

		auto var_type = type_desc::to_vtype(*desc);

		if (var_type == vtype::vec2 || var_type == vtype::vec3) {

			const static type_desc::info decimal_singleton = type_desc::decimal{};

			if (member == "x" || member == "y") return &decimal_singleton;
			if (member == "z" && var_type == vtype::vec3) {
				return &decimal_singleton;
			}

			// unknown field on vec2/vec3
			return semantic_error::make<"unknown member `{}` on type `{}`">(node, member, var_type);
		}

		if (var_type == vtype::group) {
			const auto& group = std::get<type_desc::group>(*desc);

			if (auto info = group.members.find(member); info != group.members.end()) {
				return &info->second;
			}

			// unknown field
			return semantic_error::make<"unknown member `{}` on group `{}`">(node, member, var);
		}

		// cannot index this type
		return semantic_error::make<"variable `{}` is not an aggregate type">(node, node->content());
	}

	return semantic_error::make<"unknown error">(node);
}

std::expected<flang::vtype, flang::semantic_error> flang::type_of_expression(const ast_node* node, const var_definitions& globals, const scope_stack& scopes) {

	if (auto type = detail::basic_type_map().find(node->type()); type != detail::basic_type_map().end()) {
		return type->second;
	}

	if (node->is_type<grammar::variable>() || node->is_type<grammar::member_access>()) {
		const auto value = resolve(node, globals, scopes);

		if (not value) {
			// need to pass this through
			return std::unexpected{ value.error() };
		}

		return type_desc::to_vtype(*value.value());
	}

	if (node->is_type<grammar::op::un_b_not>()) {
		const auto value = type_of_expression(node->nth(0), globals, scopes);

		if (not value) {
			return std::unexpected{ value.error() };
		}

		if (value.value() == vtype::boolean) {
			return vtype::boolean;
		}

		return semantic_error::make<"cannot apply unary operation `{}` to type `{}`">(node, node->type(), value.value());
	}

	if (node->is_type<grammar::op::un_negative>()) {
		const auto value = type_of_expression(node->nth(0), globals, scopes);

		if (not value) {
			return std::unexpected{ value.error() };
		}

		if (auto vt = value.value(); vt == vtype::decimal || vt == vtype::integer || vt == vtype::vec2 || vt == vtype::vec3) {
			return vt;
		}

		return semantic_error::make<"cannot apply unary operation `{}` to type `{}`">(node, node->type(), value.value());
	}

	if (auto op = detail::op_type_map().find(node->type()); op != detail::op_type_map().end()) {
		const auto& overloads = op->second;

		const auto lhs = type_of_expression(node->nth(0), globals, scopes);

		if (not lhs) {
			return std::unexpected{ lhs.error() };
		}

		const auto rhs = type_of_expression(node->nth(1), globals, scopes);

		if (not rhs) {
			return std::unexpected{ rhs.error() };
		}

		const auto key = std::pair{ *lhs, *rhs };
		if (auto result = overloads.find(key); result != overloads.end()) {
			return result->second;
		}

		return semantic_error::make<"incompatible types (`{}`, `{}`) for operation `{}`">(node, key.first, key.second, node->type());
	}

	if (node->is_type<grammar::expr::parenthesized>()) {
		return type_of_expression(node->nth(0), globals, scopes);
	}

	if (node->is_type<grammar::expr::call>()) {

		const auto info = resolve(node->nth(0), globals, scopes);

		if (not info) {
			// passthrough error from resolve
			return std::unexpected{ info.error() };
		}

		if (auto type = type_desc::to_vtype(*info.value()); type != vtype::function) {
			// type is not a function
			return semantic_error::make<"variable is not a function, actual type `{}`">(node, type);
		}

		const auto& func = std::get<type_desc::function>(*info.value());

		std::vector<vtype> args = {};
		for (int arg = 1; arg < node->size(); arg++) {
			auto type = type_of_expression(node->nth(arg), globals, scopes);
			if (not type) {
				return std::unexpected{ type.error() };
			}
			args.push_back(*type);
		}

		auto find_overload = [](const type_desc::function& func, const std::vector<vtype>& args) -> std::optional<vtype> {
			for (const auto& overload : func.overloads) {
				if (overload.args.size() != args.size()) continue;

				if (std::equal(overload.args.begin(), overload.args.end(), args.begin())) {
					return overload.return_type;
				}
			}

			return std::nullopt;
		};

		if (auto overload = find_overload(func, args); overload) {
			return *overload;
		}

		std::vector<vtype> decimal_args = args;
		for (auto& arg : decimal_args) {
			if (arg == vtype::integer) arg = vtype::decimal;
		}

		if (auto overload = find_overload(func, decimal_args); overload) {
			return *overload;
		}

		std::string str_args = "(";
		for (int i = 0; i < args.size(); i++) {
			str_args += vtype_to_string(args[i]);
			if (i + 1 == args.size()) {
				str_args += ")";
			}
			else str_args += ", ";
		}
		// no overload exists
		return std::unexpected{
			semantic_error{ node,std::format("cannot find suitable overload, types are {}", str_args) } };
	}

	// cannot determine type of expression
	return std::unexpected{ semantic_error{ node, "unknown error" } };
}

bool flang::is_referenced(const ast_node* n, std::string_view name, const scope_stack& scopes) {
	if (n->is_type<grammar::variable>()
		&& n->content() == name
		&& not (n->parent()->is_type<grammar::declaration_statement>() && n->parent()->nth(0) == n)) {
		return true;
	}

	for (const auto* c : *n) {
		if (is_referenced(c, name, scopes)) {
			return true;
		}
	}

	return false;
}

bool flang::var_exists(const std::string& name, const var_definitions& globals, const scope_stack& scopes) {
	if (globals.members.contains(name)) return true;
	for (const auto& scope : scopes) {
		if (scope.members.contains(name)) return true;
	}

	return false;
}

std::optional<flang::semantic_error> flang::validate(const ast_node* n, const var_definitions& globals, scope_stack& scopes) {
	if (n->type() == "root" || n->is_type<grammar::scoped>()) {
		scopes.emplace_front();
		for (const auto& c : *n) {
			if (auto err = validate(c, globals, scopes); err) {
				return err;
			}
		}
		scopes.pop_front();
		return std::nullopt;
	}

	if (n->is_type<grammar::declaration_statement>()) {
		const auto& name = n->nth(0)->content();

		if (var_exists(name, globals, scopes)) {
			return semantic_error{ n, std::format("variable `{}` is already declared", name) };
		}

		if (n->parent()->is_type<grammar::if_statement>()) {
			return semantic_error{ n, "declaration has no effect; scope is left immediately" };
		}

		auto type = type_of_expression(n->nth(1), globals, scopes);

		if (not type) {
			return type.error();
		}

		switch (type.value()) {
		case vtype::boolean: scopes.front().members[name] = type_desc::boolean{}; break;
		case vtype::decimal: scopes.front().members[name] = type_desc::decimal{}; break;
		case vtype::integer: scopes.front().members[name] = type_desc::integer{}; break;
		case vtype::vec2: scopes.front().members[name] = type_desc::vec2{}; break;
		case vtype::vec3: scopes.front().members[name] = type_desc::vec3{}; break;
		default: return semantic_error{ n, std::format("cannot declare variable of type `{}`", vtype_to_string(*type)) };
		}

		if (not is_referenced(n->parent(), name, scopes)) {
			return semantic_error{ n, std::format("variable `{}` is never used", name) };
		}

		return std::nullopt;
	}

	if (n->is_type<grammar::assignment_statement>()) {
		if (auto type = type_of_expression(n->nth(0), globals, scopes); not type) {
			return type.error();
		}
	}

	if (n->is_type<grammar::if_statement>()) {
		auto condition_type = type_of_expression(n->nth(0), globals, scopes);
		if (not condition_type) {
			return condition_type.error();
		}
		if (*condition_type != vtype::boolean) {
			return semantic_error{ n->nth(0), std::format("expression must be a boolean, actual type `{}`", vtype_to_string(condition_type.value())) };
		}

		for (int i = 1; i < n->size(); i++) {
			if (auto err = validate(n->nth(i), globals, scopes); err) {
				return err;
			}
		}

		return std::nullopt;
	}

	return std::nullopt;
}

bool flang::type_desc::group::has_member(std::string_view name) const { return members.find(name) != members.end(); }

bool flang::type_desc::group::add_boolean(std::string_view name) {
	if (has_member(name)) return false;
	members.emplace(name, boolean{});
	return true;
}

bool flang::type_desc::group::add_decimal(std::string_view name) {
	if (has_member(name)) return false;
	members.emplace(name, decimal{});
	return true;
}

bool flang::type_desc::group::add_integer(std::string_view name) {
	if (has_member(name)) return false;
	members.emplace(name, integer{});
	return true;
}

bool flang::type_desc::group::add_vec2(std::string_view name) {
	if (has_member(name)) return false;
	members.emplace(name, vec2{});
	return true;
}

bool flang::type_desc::group::add_vec3(std::string_view name) {
	if (has_member(name)) return false;
	members.emplace(name, vec3{});
	return true;
}

flang::type_desc::group::group() noexcept = default;
flang::type_desc::group::group(flang::type_desc::group&&) noexcept = default;
flang::type_desc::group& flang::type_desc::group::operator=(flang::type_desc::group&&) noexcept = default;

const flang::var_definitions& flang::stdlib()
{
	const static auto lib = []() -> flang::var_definitions {

		auto l = flang::var_definitions{};

		auto constants_group = flang::type_desc::group{};
		constants_group.add_decimal("pi");
		constants_group.add_decimal("inv_pi");

		l.members.emplace("math", std::move(constants_group));

		l.members.emplace("vec2", flang::type_desc::function{
			{{flang::vtype::decimal, flang::vtype::decimal}, flang::vtype::vec2}
		});

		l.members.emplace("vec3", flang::type_desc::function{
			{{flang::vtype::decimal, flang::vtype::decimal, flang::vtype::decimal}, flang::vtype::vec3}
		});

		const auto basic_math = std::vector<std::string>{
			"sin", "cos", "cosh", "sinh", "exp", "sqrt", "eps", "tan", "abs", "log", "log10", "log2"
		};

		for (const auto& name : basic_math) {
			l.members.emplace(name, flang::type_desc::function{
				{{flang::vtype::decimal}, flang::vtype::decimal},
				{{flang::vtype::vec2}, flang::vtype::vec2},
				{ {flang::vtype::vec3}, flang::vtype::vec3}
				});
		}
		
		const auto dual_return = std::vector<std::string>{
			"sincos", "cossin", "sincospi", "cossinpi"
		};

		for (const auto& name : dual_return) {
			l.members.emplace(name, flang::type_desc::function{
				{{flang::vtype::decimal}, flang::vtype::vec2}
			});
		}

		const auto binary = std::vector<std::string>{
			"copysign", "modf", "pow"
		};

		for (const auto& name : binary) {
			l.members.emplace(name, flang::type_desc::function{
				{{flang::vtype::decimal, flang::vtype::decimal}, flang::vtype::decimal}
			});
		}

		const auto nullary = std::vector<std::string>{
			"rand01", "randbit"
		};

		for (const auto& name : nullary) {
			l.members.emplace(name, flang::type_desc::function{
				{{}, flang::vtype::decimal}
			});
		}

		l.members.emplace("elmul", flang::type_desc::function{
			{{flang::vtype::vec2, flang::vtype::vec2}, flang::vtype::vec2}
		});

		l.members.emplace("randgauss", flang::type_desc::function{
			{{}, flang::vtype::vec2}
		});

		l.members.emplace("floor", flang::type_desc::function{
			{{flang::vtype::decimal}, flang::vtype::integer}
		});

		l.members.emplace("trunc", flang::type_desc::function{
			{{flang::vtype::decimal}, flang::vtype::integer}
		});

		l.members.emplace("is_even", flang::type_desc::function{ {{flang::vtype::integer}, flang::vtype::boolean } });
		l.members.emplace("is_odd", flang::type_desc::function{ {{flang::vtype::integer}, flang::vtype::boolean } });

		return l;

	}();

	return lib;
}
