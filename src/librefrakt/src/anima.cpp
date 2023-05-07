#include <librefrakt/anima.h>

rfkt::function_table::function_table() {
	vm.open_libraries(sol::lib::math);

	vm["math"]["copysign"] = [](double x, double y) { return std::copysign(x, y); };

	lua_sethook(vm.lua_state(), [](lua_State* L, lua_Debug*) {
		sol::state_view vm = L;
		auto instruction_count = vm["instruction_count"].get_or(0);
		if (instruction_count > 1000000) {
			SPDLOG_ERROR("Instruction count exceeded");
			luaL_error(L, "Instruction count exceeded");
		}
		vm["instruction_count"] = instruction_count + 1;
	}, LUA_MASKCOUNT, 1);
}

double rfkt::function_table::call(std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args) {
	if (!funcs.contains(name)) {
		SPDLOG_ERROR("function '{}' not found", name);
		return iv;
	}

	const auto& def_args = funcs.find(name)->second.first.args;
	if (def_args.size() != args.size()) {
		SPDLOG_ERROR("function '{}' expected {} arguments, got {}", name, def_args.size(), args.size());
		return iv;
	}

	std::vector<rfkt::anima::arg_t> call_args{};
	for (const auto& [arg_name, arg_value] : args) {
		if (!def_args.contains(arg_name)) {
			SPDLOG_ERROR("function '{}' does not have argument '{}'", name, arg_name);
			return iv;
		}

		auto arg_type = def_args.find(arg_name)->second;

		if (arg_type.first == func_info::arg_t::decimal && !std::holds_alternative<double>(arg_value)) {
			SPDLOG_ERROR("function '{}' argument '{}' is not a decimal", name, arg_name);
			return iv;
		}

		if (arg_type.first == func_info::arg_t::integer && !std::holds_alternative<int>(arg_value)) {
			SPDLOG_ERROR("function '{}' argument '{}' is not an integer", name, arg_name);
			return iv;
		}

		if (arg_type.first == func_info::arg_t::boolean && !std::holds_alternative<bool>(arg_value)) {
			SPDLOG_ERROR("function '{}' argument '{}' is not a boolean", name, arg_name);
			return iv;
		}

		call_args.emplace_back(arg_value);
	}
	vm["instruction_count"] = 0;
	SPDLOG_DEBUG("Calling {}", name);
	auto func = sol::protected_function{ vm[funcs.find(name)->second.second] };
	auto result = func(t, iv, sol::as_args(call_args));
	SPDLOG_DEBUG("Done calling, {} instructions", vm["instruction_count"].get_or(0));
	if (!result.valid()) {
		SPDLOG_ERROR("function '{}' failed:\n{}", name, result.get<sol::error>().what());
		return iv;
	}

	return result.get<double>();
}

std::string rfkt::function_table::create_function_source(std::string_view func_name, const func_info& fi) {
	std::string args = {};
	int i = 0;
	for (const auto& arg : fi.args) {
		args += arg.first;
		if (i + 1 != fi.args.size()) args += ", ";
		i++;
	}

	return std::format("function {}(t, iv, {})\n{}\nend", func_name, args, fi.source);
}
