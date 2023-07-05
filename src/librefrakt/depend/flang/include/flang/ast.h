#pragma once

#include <expected>
#include <list>

#include "flang/matcher.h"
#include "flang/ast_node.h"

namespace flang {

	struct position {
		std::size_t byte;
		std::size_t line;
		std::size_t column;
	};

	class parse_error {
	public:

		parse_error(position pos, std::string_view msg) :
			pos_(pos), message_(msg) {}

		const auto& pos() const { return pos_; }
		std::string_view what() const { return message_; }

	private:
		position pos_;
		std::string message_;
	};


	class ast {
	public:
		ast() = default;
		~ast() = default;

		ast(const ast&) = delete;
		ast& operator=(const ast&) = delete;

		ast(ast&&) noexcept = default;
		ast& operator=(ast&&) noexcept = default;

		const auto* head() const { return &nodes_.front(); }

		static std::expected<ast, parse_error> parse_expression(std::string_view source);
		static std::expected<ast, parse_error> parse_statement(std::string_view source);

		auto begin() const { return nodes_.cbegin(); }
		auto end() const { return nodes_.end(); }

	private:

		std::list<ast_node> nodes_;
	};

}