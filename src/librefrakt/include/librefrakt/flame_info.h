#pragma once

#include <string>
#include <set>
#include <optional>
#include <map>

#include <ranges>

#include <flang/ast.h>
#include <librefrakt/traits/hashable.h>

namespace rfkt {

	class vardata;

	class flamedb {
	public:

		struct parameter: public traits::hashable {
			std::string name;
			double default_value;
			std::set<std::string, std::less<>> tags;

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(name);
				for(const auto& tag: tags) hs.update(tag);
			}
		};

		struct variation: public traits::hashable {
			std::string name;

			std::string source;
			std::optional<std::string> precalc_source;

			std::set<std::string, std::less<>> tags;
			std::map<std::string, parameter, std::less<>> parameters;
	
			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(name);
				hs.update(source);
				if (precalc_source) {
					hs.update(precalc_source.value());
				}

				for(const auto& [_, param]: parameters) {
					param.add_to_hash(hs);
				}
			}
		};

		struct variation_ast {
			const flang::ast* apply;
			const flang::ast* precalc;
		};

		bool is_variation(std::string_view name) const noexcept {
			return variations_.contains(name);
		}

		bool is_parameter(std::string_view vname, std::string_view pname) const noexcept {
			if (not variations_.contains(vname)) return false;
			return variations_.find(vname)->second.parameters.contains(pname);
		}

		bool is_common(std::string_view name) const noexcept {
			return common_.contains(name);
		}

		auto variations() const noexcept {
			return std::views::values(variations_);
		}

		auto common() const noexcept {
			return std::views::all(common_);
		}

		bool add_or_update_variation(const variation&) noexcept;
		bool add_or_update_common(std::string_view name, std::string_view code) noexcept;

		const variation& get_variation(std::string_view name) const noexcept {
			return variations_.find(name)->second;
		}

		const std::string& get_common(std::string_view name) const noexcept {
			return common_.find(name)->second;
		}

		variation_ast get_variation_ast(std::string_view name) const noexcept {
			const auto cache = variation_cache_.find(name);
			return variation_ast{
				&cache->second.first,
				cache->second.second ? &cache->second.second.value() : nullptr
			};
		}

		const flang::ast& get_common_ast(std::string_view name) const noexcept {
			return common_cache_.find(name)->second;
		}

		bool remove_variation(std::string_view name) noexcept {
			if (variations_.erase(name) > 0) {
				variation_cache_.erase(name);
				recalc_hash();
				return true;
			}
			return false;
		}

		bool remove_common(std::string_view name) noexcept {
			if (common_.erase(name) > 0) {
				common_cache_.erase(name);
				recalc_hash();
				return true;
			}
			return false;
		}

		auto hash() const noexcept {
			return hash_;
		}

		auto make_vardata(std::string_view vname) const noexcept -> std::pair<std::string, vardata>;

	private:

		void recalc_hash() noexcept;

		rfkt::hash_t hash_;

		std::map<std::string, variation, std::less<>> variations_;
		std::map<std::string, std::string, std::less<>> common_;

		using cache_value_type = std::pair<flang::ast, std::optional<flang::ast>>;
		std::map<std::string, cache_value_type, std::less<>> variation_cache_;
		std::map<std::string, flang::ast, std::less<>> common_cache_;
	};

	void initialize(rfkt::flamedb& fdb, std::string_view config_path);

}