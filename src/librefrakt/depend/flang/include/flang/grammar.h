#pragma once

#include <variant>
#include <map>
#include <vector>
#include <expected>
#include <format>
#include <optional>
#include <list> 

#include "flang/ast_node.h"
#include "flang/matcher.h"

namespace flang::grammar {

	namespace lit {
		struct integer;
		struct decimal;
		struct boolean;
	}

	namespace op {
		struct plus;
		struct minus;
		struct times;
		struct divided;
		struct exponent;

		struct equals;
		struct not_equals;
		struct less_than;
		struct less_equal;
		struct greater_than;
		struct greater_equal;

		struct b_and;
		struct b_or;

		struct un_b_not;
		struct un_negative;

		struct assignment;
		struct plus_assign;
		struct minus_assign;
		struct times_assign;
		struct divided_assign;

		struct member;
	}

	namespace expr {
		struct parenthesized;
		struct call;
	}

	struct variable;
	struct member_access;

	struct expression;
	struct declaration;
	struct if_statement;
	struct assignment_statement;
	struct declaration_statement;
	struct break_statement;

	struct scoped;

	template<typename T>
	consteval std::string_view demangle() noexcept {
		return tao::pegtl::demangle<T>();
	}
}

namespace flang {

	enum class vtype {
		boolean,
		decimal,
		integer,
		vec2,
		vec3,
		group,
		function
	};

	constexpr std::string_view vtype_to_string(vtype v) {
		switch (v) {
		case vtype::boolean: return "boolean";
		case vtype::decimal: return "decimal";
		case vtype::integer: return "integer";
		case vtype::vec2: return "vec2";
		case vtype::vec3: return "vec3";
		case vtype::group: return "group";
		case vtype::function: return "function";
		}

		std::unreachable();
	}

	namespace type_desc {

		struct boolean;
		struct decimal;
		struct integer;
		struct vec2;
		struct vec3;
		struct group;
		struct function;

		using info = std::variant<type_desc::boolean, type_desc::decimal, type_desc::integer, type_desc::vec2, type_desc::vec3, type_desc::group, type_desc::function>;

		struct boolean { static constexpr auto type = vtype::boolean; };
		struct decimal { static constexpr auto type = vtype::decimal; };
		struct integer { static constexpr auto type = vtype::integer; };
		struct vec2 { static constexpr auto type = vtype::vec2; };
		struct vec3 { static constexpr auto type = vtype::vec3; };

		struct group {
			std::map<std::string, info, std::less<>> members;

			group() noexcept;
			group(group&&) noexcept;
			group& operator=(group&&) noexcept;

			bool has_member(std::string_view name) const;

			bool add_boolean(std::string_view name);
			bool add_decimal(std::string_view name);
			bool add_integer(std::string_view name);
			bool add_vec2(std::string_view name);
			bool add_vec3(std::string_view name);

			static constexpr auto type = vtype::group;
		};

		struct function {
			struct signature {
				std::vector<vtype> args;
				vtype return_type;
			};

			explicit function(std::initializer_list<signature> funcs) : overloads(funcs) {}

			std::vector<signature> overloads;

			static constexpr auto type = vtype::function;
		};

		inline vtype to_vtype(const info& i) {
			return std::visit([]<typename T>(const T&) {
				return T::type;
			}, i);
		}
	};

	using var_definitions = type_desc::group;
	using scope_stack = std::list<var_definitions>;

	const var_definitions& stdlib();

	namespace detail {

		template<std::size_t Length>
		struct StringLiteral2 {
			constexpr static std::size_t length = Length;

			explicit(false) constexpr StringLiteral2(const char(&str)[Length]) {
				std::copy_n(str, Length, value);
			}

			char value[Length];
		};

		using operator_map = std::map<std::string_view, std::map<std::pair<vtype, vtype>, vtype>, std::less<>>;
		const operator_map& op_type_map();

		inline const std::map<std::string_view, vtype>& basic_type_map();
	}

	class semantic_error {
	public:

		semantic_error() = delete;

		semantic_error(const ast_node* ref, std::string_view msg) :
			node(ref),
			message(msg) {}

		const ast_node* ref() const { return node; }
		std::string_view what() const { return message; }

		template<detail::StringLiteral2 format_string, typename... Args>
		static std::unexpected<semantic_error> make(const ast_node* ref, Args&&... args) {
			return std::unexpected{ semantic_error{ref, std::format(format_string.value, std::forward<Args>(args)...)} };
		}

	private:
		const ast_node* node;
		std::string message;
	};

}

template<>
struct std::formatter<flang::vtype> {
	constexpr auto parse(const std::format_parse_context& ctx) const {
		return ctx.begin();
	}

	auto format(const flang::vtype& vt, std::format_context& ctx) {
		return std::format_to(ctx.out(), "{}", flang::vtype_to_string(vt));
	}
};

namespace flang {

	std::expected<const type_desc::info*, semantic_error> resolve(const ast_node* node, const var_definitions& globals, const scope_stack& scopes);
	std::expected<vtype, semantic_error> type_of_expression(const ast_node* node, const var_definitions& globals, const scope_stack& scopes);
	bool is_referenced(const ast_node* n, std::string_view name, const scope_stack& scopes);
	bool var_exists(const std::string& name, const var_definitions& globals, const scope_stack& scopes);;
	std::optional<semantic_error> validate(const ast_node* n, const var_definitions& globals, scope_stack& scopes);
}