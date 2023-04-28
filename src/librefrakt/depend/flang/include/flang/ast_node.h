#pragma once

#include <vector>
#include <string>

#include <tao/pegtl/demangle.hpp>

namespace flang {

	class ast_node {
	public:

		auto begin() const noexcept { return children_.cbegin(); }
		auto end() const noexcept { return children_.cend(); }

		auto first() const noexcept -> const ast_node* { return children_.front(); }
		auto last() const noexcept -> const ast_node* { return children_.back(); }

		auto nth(std::size_t n) const noexcept -> const ast_node* { return children_.at(n); }

		auto size() const noexcept { return children_.size(); }
		auto parent() const noexcept { return parent_; }

		const std::string& type() const noexcept { return type_; }
		const std::string& content() const noexcept { return content_; }

		template<typename T>
		bool is_type() const noexcept { return tao::pegtl::demangle<T>() == type_; }

		ast_node(std::string_view type, std::string_view content, ast_node* parent, std::vector<ast_node*>&& children) noexcept :
			type_(std::string{ type }), content_(std::string{ content }), parent_(parent), children_(std::move(children)) {}

		ast_node() = default;

		template<typename Pred>
		std::vector<const ast_node*> find_descendents(Pred&& pred) const noexcept {
			std::vector<const ast_node*> result;
			for (const auto& child : *this) {
				if (pred(child)) {
					result.push_back(child);
				}
				auto sub = child->find_descendants(pred);
				result.insert(result.end(), sub.begin(), sub.end());
			}
			return result;
		}

		template<typename Pred>
		bool has_descendent(Pred&& pred) const noexcept {
			for (const auto& child : *this) {
				if (pred(child)) {
					return true;
				}
				if (child->has_descendent(pred)) {
					return true;
				}
			}
			return false;
		}

	private:

		friend class ast;
		friend class node_converter;

		std::string type_;
		std::string content_;

		ast_node* parent_ = nullptr;
		std::vector<ast_node*> children_;

	};
}