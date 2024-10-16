#pragma once

#include <algorithm>
#include <tao/pegtl/demangle.hpp>

#include "flang/ast_node.h"

namespace flang {

	using matcher = std::add_pointer_t<bool(const ast_node* n)>;

	namespace detail {

		template<typename T>
		concept matcher_concept = std::is_empty_v<T> && requires(T, const ast_node* ptr) {
			{ T{}(ptr) } -> std::same_as<bool>;
		};

		template<size_t N>
		struct string_literal {
			constexpr string_literal(const char(&str)[N]) {
				std::copy_n(str, N, value);
			}

			char value[N];
		};
	}

}

template<flang::detail::matcher_concept L, flang::detail::matcher_concept R>
[[nodiscard]] consteval auto operator&&(L, R) noexcept {
	return [](const flang::ast_node* node) -> bool {
		return L{}(node) && R{}(node);
	};
}

template<flang::detail::matcher_concept L, flang::detail::matcher_concept R>
[[nodiscard]] consteval auto operator||(L, R) noexcept {
	return [](const flang::ast_node* node) -> bool {
		return L{}(node) || R{}(node);
	};
}

template<flang::detail::matcher_concept M>
[[nodiscard]] consteval auto operator!(M) noexcept {
	return [](const flang::ast_node* node) -> bool {
		return !M{}(node);
	};
}

namespace flang::matchers {
	template<typename... Ts>
	constexpr static auto of_type = 
	[](const flang::ast_node* node) -> bool {
		return ((node->type() == tao::pegtl::demangle<Ts>()) || ...);
	};

	constexpr static auto is_root =
	[](const flang::ast_node* node) -> bool {
		return node->parent() == nullptr;
	};

	template<flang::detail::string_literal... Strs>
	constexpr static auto with_content =
	[](const flang::ast_node* node) -> bool {
		return ((node->content() == Strs.value) || ...);
	};

	template<int Rank>
	constexpr static auto of_rank =
	[](const flang::ast_node* node) -> bool {
		const auto* p = node->parent();
		return p && p->size() > Rank && p->nth(Rank) == node;
	};

	template<flang::detail::matcher_concept Pred>
	consteval auto with_child(Pred) noexcept {
		return [](const flang::ast_node* node) -> bool {
			return std::any_of(node->begin(), node->end(), Pred{});
		};
	}

	template<int Rank, flang::detail::matcher_concept Pred>
	consteval auto with_child_at(Pred) noexcept {
		return [](const flang::ast_node* node) -> bool {
			constexpr static auto pred = Pred{};
			return node->size() > Rank && pred(node->nth(Rank));
		};
	}

	template<flang::detail::matcher_concept Pred>
	consteval auto with_parent(Pred) noexcept {
		return [](const flang::ast_node* node) -> bool {
			if (node->parent() == nullptr) {
				return false;
			}
			constexpr static auto pred = Pred{};
			return pred(node->parent());
		};
	}

	template<flang::detail::matcher_concept Pred>
	consteval auto with_sibling(Pred) noexcept {
		return [](const flang::ast_node* node) -> bool {
			constexpr static auto pred = Pred{};
			if (node->parent() == nullptr) {
				return false;
			}

			for (const auto* s : *node->parent()) {
				if (s != node && pred(s)) {
					return true;
				}
			}

			return false;
		};
	}
}