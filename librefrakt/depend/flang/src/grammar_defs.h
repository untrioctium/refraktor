#pragma once

#include <tao/pegtl.hpp>
#include <tao/pegtl/demangle.hpp>

namespace peg = tao::pegtl;

namespace flang::grammar {

	struct statement;

	template<char... Symbols>
	using symbol = peg::pad<peg::string<Symbols...>, peg::space>;

	namespace lit {
		struct integer : peg::seq<peg::opt<peg::one<'+', '-'>>, peg::plus<peg::digit>> {};
		struct decimal : peg::seq<peg::opt<peg::one<'+', '-'>>, peg::plus<peg::digit>, peg::one<'.'>, peg::plus<peg::digit>> {};
		struct boolean : peg::pad<peg::sor< TAO_PEGTL_STRING("true"), TAO_PEGTL_STRING("false")>, peg::space> {};
	}

	struct literal : peg::sor<lit::decimal, lit::integer, lit::boolean> {};

	namespace op {

		// mathematical operators
		struct plus : symbol<'+'> {};
		struct minus : symbol<'-'> {};
		struct times : symbol<'*'> {};
		struct divided : symbol<'/'> {};
		struct exponent : symbol<'*', '*'> {};

		// comparison
		struct equals : symbol<'=', '='> {};
		struct not_equals : symbol<'!', '='> {};
		struct less_than : symbol<'<'> {};
		struct less_equal : symbol<'<', '='> {};
		struct greater_than : symbol<'>'> {};
		struct greater_equal : symbol<'>', '='> {};

		struct comparison_operators : peg::sor<op::less_than, op::less_equal, op::greater_than, op::greater_equal> {};

		// boolean
		struct b_and : peg::sor<symbol<'&', '&'>, TAO_PEGTL_STRING("and")> {};
		struct b_or : peg::sor<symbol<'|', '|'>, TAO_PEGTL_STRING("or")> {};

		// unary operators, no padding allowed with these
		struct un_negative : peg::one<'-'> {};
		struct un_b_not : peg::one<'!'> {};

		// assignment and compound assignment
		struct assignment : symbol<'='> {};
		struct plus_assign : symbol < '+', '=' > {};
		struct minus_assign : symbol < '-', '=' > {};
		struct times_assign : symbol < '*', '=' > {};
		struct divided_assign : symbol < '/', '=' > {};

		struct assignment_operators : peg::sor<assignment, plus_assign, minus_assign, times_assign, divided_assign> {};

		// misc
		struct member : peg::one<'.'> {};

	}

	namespace syn {
		struct semi : symbol<';'> {};
		struct colon : symbol<':'> {};
		struct conditional : symbol<'?'> {};
		struct comma : symbol<','> {};
		struct declaration : symbol<':', '='> {};

		struct open_paren : symbol<'('> {};
		struct close_paren : symbol<')'> {};

		struct open_brace : symbol<'{'> {};
		struct close_brace : symbol<'}'> {};

		struct token_if : symbol<'i', 'f'> {};
		struct token_else : symbol<'e', 'l', 's', 'e'> {};
	}

	struct variable : peg::identifier {};
	struct member_access : peg::seq<variable, op::member, variable> {};
	struct lvalue : peg::sor<member_access, variable> {};

	namespace expr {
		struct parenthesized : peg::if_must< syn::open_paren, expression, syn::close_paren > {};
		struct call : peg::if_must<peg::seq<lvalue, syn::open_paren>, peg::opt<peg::list_must<expression, syn::comma>>, syn::close_paren> {};
		struct value : peg::sor< literal, call, lvalue, parenthesized > {};
		struct unary : peg::seq< peg::opt<peg::sor<op::un_negative, op::un_b_not>>, value > {};
		struct power : peg::list_must< unary, op::exponent > {};
		struct product : peg::list_must< power, peg::sor< op::times, op::divided >> {};
		struct sum : peg::list_must< product, peg::sor< op::plus, op::minus >> {};
		struct comparison : peg::list_must< sum, op::comparison_operators> {};
		struct equality : peg::list_must<comparison, peg::sor<op::equals, op::not_equals>> {};
		struct boolean : peg::list_must<equality, peg::sor<op::b_and, op::b_or>> {};
		struct ternary : peg::sor< peg::seq<syn::open_paren, boolean, syn::close_paren, syn::conditional, boolean, syn::colon, boolean>, boolean> {};
		struct assignment : peg::seq<lvalue, op::assignment_operators, expression> {};
	}

	struct expression : expr::ternary {};
	struct declaration : peg::if_must<peg::seq<lvalue, syn::declaration>, expression> {};

	struct if_statement : peg::if_must<syn::token_if, peg::pad<expression, peg::space>, statement, peg::opt_must<syn::token_else, statement>> {};

	struct assignment_statement : peg::if_must<expr::assignment, syn::semi> {};
	struct declaration_statement : peg::if_must<declaration, syn::semi> {};

	struct statement : peg::pad<peg::sor<if_statement, assignment_statement, declaration_statement, scoped>, peg::space>{};
	struct scoped : peg::if_must<syn::open_brace, peg::must<peg::plus<statement>>, syn::close_brace> {};

	struct body : peg::must<peg::plus<statement>, peg::eof> {};
}