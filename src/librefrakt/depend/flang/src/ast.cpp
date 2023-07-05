#include "flang/grammar.h"
#include "flang/ast.h"

#include "grammar_defs.h"

#include <tao/pegtl/contrib/parse_tree.hpp>

struct rearrange
	: peg::parse_tree::apply< rearrange > {
	template< typename Node, typename... States >
	static void transform(std::unique_ptr< Node >& n, States&&... st)
	{
		if (n->children.size() == 1) {
			n = std::move(n->children.back());
		}
		else if (n->children.size() == 2) {
			n->remove_content();
			auto& c = n->children;
			auto v = std::move(c.back());
			c.pop_back();
			auto o = std::move(c.back());
			o->children.emplace_back(std::move(v));
			n = std::move(o);
		}
		else if(n->children.size() >= 3) {
			n->remove_content();
			auto& c = n->children;
			auto r = std::move(c.back());
			c.pop_back();
			auto o = std::move(c.back());
			c.pop_back();
			o->children.emplace_back(std::move(n));
			o->children.emplace_back(std::move(r));
			n = std::move(o);
			transform(n->children.front(), st...);
		}
	}
};

template<typename Rule>
using selector = peg::parse_tree::selector <
	Rule,
	peg::parse_tree::store_content::on<
	flang::grammar::lit::boolean,
	flang::grammar::lit::decimal,
	flang::grammar::lit::integer,
	flang::grammar::variable
	>,
	peg::parse_tree::remove_content::on <
	flang::grammar::op::plus,
	flang::grammar::op::minus,
	flang::grammar::op::times,
	flang::grammar::op::divided,
	flang::grammar::op::exponent,
	flang::grammar::op::equals,
	flang::grammar::op::not_equals,
	flang::grammar::op::less_than,
	flang::grammar::op::less_equal,
	flang::grammar::op::greater_than,
	flang::grammar::op::greater_equal,
	flang::grammar::op::b_and,
	flang::grammar::op::b_or,
	flang::grammar::op::un_b_not,
	flang::grammar::op::un_negative,
	flang::grammar::op::assignment,
	flang::grammar::op::plus_assign,
	flang::grammar::op::minus_assign,
	flang::grammar::op::times_assign,
	flang::grammar::op::divided_assign,
	flang::grammar::syn::conditional,
	flang::grammar::expr::call,
	flang::grammar::expr::parenthesized,
	flang::grammar::member_access,
	flang::grammar::scoped,
	flang::grammar::if_statement,
	flang::grammar::assignment_statement,
	flang::grammar::declaration_statement,
	flang::grammar::break_statement
	>,
	rearrange::on<
	flang::grammar::expr::unary, 
	flang::grammar::expr::power, 
	flang::grammar::expr::product, 
	flang::grammar::expr::sum, 
	flang::grammar::expr::comparison, 
	flang::grammar::expr::equality, 
	flang::grammar::expr::boolean, 
	flang::grammar::expr::assignment>
> ;

using peg_node = std::unique_ptr<tao::pegtl::parse_tree::node>;

std::size_t count_nodes(const peg_node& n) {

	std::size_t size = 1;

	for (const auto& c : n->children) {
		size += count_nodes(c);
	}

	return size;
}

template<typename Func>
concept node_allocator = requires(Func&& f) {
	{f()} -> std::same_as<flang::ast_node*>;
};
namespace flang {
	class node_converter {
	public:
		static flang::ast_node* convert_recursive(const peg_node& n, flang::ast_node* parent, node_allocator auto&& nalloc) {
			flang::ast_node* astn = nalloc();

			astn->parent_ = parent;
			astn->type_ = (parent != nullptr) ? n->type : "root";
			if (n->has_content()) astn->content_ = n->string();

			for (const auto& c : n->children) {
				astn->children_.push_back(convert_recursive(c, astn, nalloc));
			}

			return astn;
		}

	};
}
template<typename T>
std::optional<flang::parse_error> parse_impl(std::list<flang::ast_node>& nodes, std::string_view source) {
	auto input = peg::string_input(source, "");

	try {
		auto root = peg::parse_tree::parse<T, peg::parse_tree::node, selector>(input);

		flang::node_converter::convert_recursive(root, nullptr,
			[&nodes]() {
				nodes.emplace_back();
				return &nodes.back();
			}
		);

		return std::nullopt;
	}
	catch (const peg::parse_error& e) {
		auto pos = flang::position{ 0,0,0 };
		if (e.positions().size()) {
			const auto& peg_pos = e.positions()[0];
			pos.byte = peg_pos.byte;
			pos.line = peg_pos.line;
			pos.column = peg_pos.column;
		}
		return flang::parse_error{pos, e.message() };
	}
}

std::expected<flang::ast, flang::parse_error> flang::ast::parse_statement(std::string_view source)
{
	auto tree = ast{};
	auto err = parse_impl<grammar::body>(tree.nodes_, source);

	if (err) {
		return std::unexpected{ err.value() };
	}
	return tree;
}

std::expected<flang::ast, flang::parse_error> flang::ast::parse_expression(std::string_view source)
{
	auto tree = ast{};

	using expression_t = peg::must<grammar::expression, peg::opt<grammar::syn::semi>, peg::eof>;
	auto err = parse_impl<expression_t>(tree.nodes_, source);

	if (err) {
		return std::unexpected{ err.value() };
	}
	return tree;
}