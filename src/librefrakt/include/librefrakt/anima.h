#pragma once

#include <variant>
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>
#include <ranges>

#include <librefrakt/flame_types.h>

namespace rfkt {

	struct func_info {
		enum class arg_t {
			decimal,
			integer,
			boolean
		};

		std::map<std::string, std::pair<arg_t, anima::arg_t>> args;
		std::string source;
	};

	class function_table {
	public:

		function_table();

		bool add_or_update(std::string_view name, func_info&& info) {
			auto name_hash = std::format("af_{}", rfkt::hash::calc(name).str32());
			auto source = create_function_source(name_hash, info);
			auto result = vm.safe_script(source, sol::script_throw_on_error);
			if (!result.valid()) {
				SPDLOG_ERROR("failed to compile function '{}': {}", name, result.get<sol::error>().what());
				return false;
			}

			funcs.emplace(name, std::pair{ std::move(info), std::move(name_hash) });
		}

		double call(std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args);

		rfkt::anima::call_info_t make_default(std::string_view name) const noexcept {
			if(!funcs.contains(name)) return std::nullopt;

			const auto& def_args = funcs.find(name)->second.first.args;
			auto ret = std::make_optional(rfkt::anima::call_info_t::value_type{});
			ret->first = std::string{ name };
			for (const auto& [arg_name, arg_type] : def_args) {
				ret->second.emplace(arg_name, arg_type.second);
			}

			return ret;
		}

		auto names() const noexcept {
			return std::views::keys(funcs);
		}

		auto functions() const noexcept {
			return funcs;
		}
	private:

		static std::string create_function_source(std::string_view func_name, const func_info& fi);

		std::map<std::string, std::pair<func_info, std::string>, std::less<>> funcs;
		sol::state vm;
	};

}