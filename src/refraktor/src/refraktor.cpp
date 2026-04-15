
#include <cpujit.hpp>
#include <stack>
#include <tao/pegtl/demangle.hpp>
#include <flang/ast.hpp>
#include <flang/grammar.hpp>

#include <iostream>

// FNV-1a constants for 64-bit
constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;

// Constexpr 64-bit hash function
constexpr uint64_t hash_fnv1a_64(std::string_view str) noexcept {
    uint64_t hash = FNV_OFFSET_BASIS;
    for (char c : str) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        hash *= FNV_PRIME;
    }
    return hash;
}

template<typename T>
constexpr static uint64_t name_hash = hash_fnv1a_64(tao::pegtl::demangle<T>());

void dump_ast(const flang::ast_node* node, int depth = 0) {

	auto depth_str = std::string(depth, ' ');
	std::cout << depth_str << node->type() << " " << node->content() << std::endl;
	for (const auto child : *node) {
		dump_ast(child, depth + 1);
	}
}

struct emit_context {

	emit_context(std::ostream& os) : os(os) {}

	// where to emit LLVM IR
	std::ostream& os;

	// local and global variables
	std::map<std::string, flang::vtype> locals{};
	std::map<std::string, flang::vtype> globals{};


	struct reg_name_info_t {
		std::string token{};
		int count = 0;
	};

	// stack used to construct unique register names
	reg_name_info_t reg_name_info;
};

// emit functions return this type so that the caller can use the intermediate registers
// e.g. 3 + sin(5) * 4 needs several intermediate steps, so the plus emitter needs to know the names assigned to the intermediate steps it cares about
// if the name is not present, it usually means a value was not emitted (e.g. an if statement creates control flow, not values)
// the values of literals are also returned as "names"
struct emit_ret_t {
	std::string name;
	flang::vtype type;
};

std::string make_register_name(emit_context& ctx, const flang::ast_node* node) {

	if (node->parent()->is_type<flang::grammar::declaration_statement>() or node->parent()->is_type<flang::grammar::op::assignment>()) {
		return std::format("%{}", ctx.reg_name_info.token);
	}

	return std::format("%{}.{:04d}", ctx.reg_name_info.token, ++ctx.reg_name_info.count);
}


emit_ret_t emit(emit_context& ctx, const flang::ast_node* node) {
	switch (auto node_hash = hash_fnv1a_64(node->type())) {
		case hash_fnv1a_64("root"):
			for (const auto child : *node) {
				emit(ctx, child);
			}
			return {};

		case name_hash<flang::grammar::declaration_statement>: {
			auto name = node->nth(0)->content();
			ctx.reg_name_info.token = name;
			ctx.reg_name_info.count = 0;
			emit(ctx, node->nth(1));
			return {};
		}

		case name_hash<flang::grammar::op::plus>:
		case name_hash<flang::grammar::op::minus>:
		case name_hash<flang::grammar::op::times>:
		case name_hash<flang::grammar::op::divided>:
		case name_hash<flang::grammar::op::exponent>:
		{
			auto name = make_register_name(ctx, node);
			auto lt = emit(ctx, node->nth(0));
			auto rt = emit(ctx, node->nth(1));
			ctx.os << std::format("{} = {} {} {}\n", name, node->type(), lt.name, rt.name);
			return { name, lt.type };
		}

		case name_hash<flang::grammar::expr::call>:
		{
			auto name = make_register_name(ctx, node);
			auto func = emit(ctx, node->nth(0));
			std::vector<emit_ret_t> args;
			for (int i = 1; i < node->size(); i++) {
				auto arg = emit(ctx, node->nth(i));
				args.push_back(arg);
			}

			ctx.os << std::format("{} = call @{}", name, node->nth(0)->content());
			for (int i = 1; i < node->size(); i++) {
				ctx.os << std::format(" {}", args[i - 1].name);
			}
			ctx.os << "\n";

			return { name, func.type };
		}

		case name_hash<flang::grammar::lit::integer>:
			return { node->content(), flang::vtype::integer };
		case name_hash<flang::grammar::lit::decimal>:
			return { node->content(), flang::vtype::decimal };
		case name_hash<flang::grammar::lit::boolean>:
			return { node->content(), flang::vtype::boolean };

		case name_hash<flang::grammar::variable>:
			return { std::format("%{}", node->content()), flang::vtype::integer }; // TODO: proper type lookup

		default:
			ctx.os << "@" << node->type() << "@";
			return {};
	}
}

void emit(std::ostream& os, const flang::ast_node* node) {
	emit_context ctx(os);
	emit(ctx, node);
}

int main(int argc, char* argv[]) {

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <flang expression>" << std::endl;
		return 1;
	}

	std::string expression = std::string(argv[1]);
	for (int i = 2; i < argc; i++) {
		expression += " " + std::string(argv[i]);
	}

	auto ast = flang::ast::parse_statement(expression);
	if (!ast) {
		std::cerr << "Failed to parse expression: " << ast.error().what() << std::endl;
		return 1;
	}

	dump_ast(ast->head());
	std::cout << std::endl;

	emit(std::cout, ast->head());



	return 0;
}