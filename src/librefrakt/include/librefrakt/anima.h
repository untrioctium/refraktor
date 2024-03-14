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

		struct arg_info {
			arg_t type;
			anima::arg_t default_value;
		};

		std::map<std::string, arg_info, std::less<>> args;
		std::string source;
	};

	class function_table {
	public:

		function_table();

		bool add_or_update(std::string_view name, func_info&& info);

		double call(std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args) {
			return call_impl(name, t, iv, args, "");
		}

		rfkt::anima::call_info_t make_default(std::string_view name) const noexcept;

		auto names() const noexcept {
			return std::views::keys(funcs);
		}

		auto functions() const noexcept {
			return funcs;
		}

		auto make_invoker() noexcept {
			return [this](std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args) {
				return this->call(name, t, iv, args);
			};
		}

		function_table clone() const noexcept {
			function_table ft;

			for (const auto& [name, info] : funcs) {
				ft.add_or_update(name, func_info{ info.info });
			}

			return ft;
		}

	private:

		double call_impl(std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args, std::string_view prefix);

		static std::string create_function_source(std::string_view func_name, const func_info& fi);

		struct stored_info {
			func_info info;
			sol::protected_function func;
		};

		std::map<std::string, stored_info, std::less<>> funcs;
		sol::state vm;
	};

}