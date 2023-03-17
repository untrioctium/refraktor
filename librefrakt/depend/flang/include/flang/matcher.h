#pragma once

#include <functional>
#include <tao/pegtl/demangle.hpp>

#include "flang/ast_node.h"

namespace flang {

	class matcher {
	public:
		using func_t = std::function<bool(const ast_node*)>;

		explicit(false) matcher(const func_t& func) :
			func_(func) {}

		//template<typename Callable>
		//matcher(Callable func) :
		//	func_(std::move(func)) {}

		bool operator()(const ast_node* node) const noexcept {
			return func_(node);
		}

		matcher operator&&(const matcher& o) const noexcept {
			return { [lhs = func_, rhs = o.func_](const ast_node* node) -> bool {
				return lhs(node) && rhs(node);
			} };
		}

		matcher operator||(const matcher& o) const noexcept {
			return { [lhs = func_, rhs = o.func_](const ast_node* node) -> bool {
				return lhs(node) || rhs(node);
			} };
		}

		matcher operator!() const noexcept {
			return { [op = func_](const ast_node* node) -> bool {
				return !op(node);
			} };
		}

	private:

		func_t func_;
	};

	namespace matchers {

		template<typename... Ts>
		matcher of_type() {
			return { [](const ast_node* node) -> bool {
				return ((node->type() == tao::pegtl::demangle<Ts>()) || ...);
			} };
		}

		inline matcher with_content(std::string_view content) {
			return { [content = std::string{content}](const ast_node* node) {
				return node->content() == content;
			} };
		}

		inline matcher with_child(matcher predicate) {
			return { [predicate](const ast_node* node) {
				for (const auto* c : *node) {
					if (predicate(c)) return true;
				}

				return false;
			} };
		}

		inline matcher with_parent(matcher predicate) {
			return { [predicate](const ast_node* node) {
				return node->parent() != nullptr && predicate(node->parent());
			} };
		}

		inline matcher with_sibling(matcher predicate) {
			return { [predicate](const ast_node* node) {
				if (node->parent() == nullptr) return false;

				for (const auto* s : *node->parent()) {
					if (s != node && predicate(s)) return true;
				}

				return false;
			} };
		}

		inline matcher of_rank(int rank)
		{
			return { [rank](const ast_node* node) {
				auto p = node->parent();
				return p && p->size() > rank && p->nth(rank) == node;
			} };
		}

	}

}