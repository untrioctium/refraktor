#include <librefrakt/anima.h>

rfkt::function_table::function_table() {
	vm.open_libraries(sol::lib::math);

	vm["math"]["copysign"] = [](double x, double y) { return std::copysign(x, y); };

	constexpr static auto instruction_count_step = 1000;
	lua_sethook(vm.lua_state(), [](lua_State* L, lua_Debug*) {
		sol::state_view vm = L;
		auto instruction_count = vm["instruction_count"].get_or(0);
		if (instruction_count > 1000000) {
			SPDLOG_ERROR("Instruction count exceeded");
			luaL_error(L, "Instruction count exceeded");
		}
		vm["instruction_count"] = instruction_count + instruction_count_step;
	}, LUA_MASKCOUNT, instruction_count_step);
}

bool rfkt::function_table::add_or_update(std::string_view name, func_info&& info) {
	auto name_hash = std::format("af_{}", rfkt::hash::calc(name).str32());
	auto source = create_function_source(name_hash, info);
	auto result = vm.safe_script(source, sol::script_throw_on_error);
	if (!result.valid()) {
		SPDLOG_ERROR("failed to compile function '{}': {}", name, result.get<sol::error>().what());
		return false;
	}

	sol::protected_function func = vm[name_hash];

	funcs.emplace(name, stored_info{ std::move(info), func });
	return true;
}

double rfkt::function_table::call_impl(std::string_view name, double t, double iv, const rfkt::anima::arg_map_t& args, std::string_view prefix) {

	if (name == "mix") {
		double left_value = iv;

		if (auto left_func_iter = args.find(std::format("{}{}", prefix, "left.function")); left_func_iter != args.end()) {

			if (!std::holds_alternative<std::string>(left_func_iter->second)) {
				SPDLOG_ERROR("left.function is not a string");
				return iv;
			}

			left_value = call_impl(std::get<std::string>(left_func_iter->second), t, iv, args, std::format("{}{}", prefix, "left."));
		}

		auto right_value_iter = args.find(std::format("{}{}", prefix, "right.value"));
		if (right_value_iter == args.end()) {
			SPDLOG_ERROR("right.value not found");
			return iv;
		}

		if (!std::holds_alternative<double>(right_value_iter->second)) {
			SPDLOG_ERROR("right.value is not a double");
			return iv;
		}

		double right_value = std::get<double>(right_value_iter->second);

		if (auto right_func_iter = args.find(std::format("{}{}", prefix, "right.function")); right_func_iter != args.end()) {

			if (!std::holds_alternative<std::string>(right_func_iter->second)) {
				SPDLOG_ERROR("right.function is not a string");
				return iv;
			}

			right_value = call_impl(std::get<std::string>(right_func_iter->second), t, right_value, args, std::format("{}{}", prefix, "right."));
		}

		double start_time = 0;
		if (auto start_time_iter = args.find(std::format("{}{}", prefix, "start_time")); start_time_iter != args.end()) {
			if (!std::holds_alternative<double>(start_time_iter->second)) {
				SPDLOG_ERROR("start_time is not a double");
				return iv;
			}

			start_time = std::get<double>(start_time_iter->second);
		}
		else {
			SPDLOG_ERROR("start_time not found");
			return iv;
		}

		double length = 0;
		if (auto length_iter = args.find(std::format("{}{}", prefix, "length")); length_iter != args.end()) {
			if (!std::holds_alternative<double>(length_iter->second)) {
				SPDLOG_ERROR("length is not a double");
				return iv;
			}

			length = std::get<double>(length_iter->second);
		}
		else {
			SPDLOG_ERROR("length not found");
			return iv;
		}

		if(t <= start_time) return left_value;
		if(t >= start_time + length) return right_value;

		double mix = (t - start_time) / length;
		mix = 6 * mix * mix * mix * mix * mix - 15 * mix * mix * mix * mix + 10 * mix * mix * mix;
		return left_value * (1 - mix) + right_value * mix;
	}

	if (!funcs.contains(name)) {
		SPDLOG_ERROR("function '{}' not found", name);
		return iv;
	}

	const auto& vm_info = funcs.find(name)->second;
	const auto& def_args = vm_info.info.args;

	std::vector<rfkt::anima::arg_t> call_args{};
	for (const auto& [arg_name, arg_type] : def_args) {
		std::string full_arg_name = std::format("{}{}", prefix, arg_name);
		auto arg_value_iter = args.find(full_arg_name);

		if (arg_value_iter == args.end()) {
			SPDLOG_ERROR("function '{}' argument '{}' not found (searched '{}')", name, arg_name, full_arg_name);
			return iv;
		}

		auto arg_value = arg_value_iter->second;

		if (arg_type.type == func_info::arg_t::decimal && !std::holds_alternative<double>(arg_value)) {
			SPDLOG_ERROR("function '{}' argument '{}' is not a decimal", name, arg_name);
			return iv;
		}

		if (arg_type.type == func_info::arg_t::integer && !std::holds_alternative<int>(arg_value)) {
			SPDLOG_ERROR("function '{}' argument '{}' is not an integer", name, arg_name);
			return iv;
		}

		if (arg_type.type == func_info::arg_t::boolean && !std::holds_alternative<bool>(arg_value)) {
			SPDLOG_ERROR("function '{}' argument '{}' is not a boolean", name, arg_name);
			return iv;
		}

		call_args.emplace_back(arg_value);
	}

	if (call_args.size() != def_args.size()) {
		SPDLOG_ERROR("function '{}' argument count mismatch, got {} expected {}", name, call_args.size(), def_args.size());
		return iv;
	}

	vm["instruction_count"] = 0;
	SPDLOG_DEBUG("Calling {}", name);
	auto result = vm_info.func(t, iv, sol::as_args(call_args));
	SPDLOG_DEBUG("Done calling, {} instructions", vm["instruction_count"].get_or(0));
	if (!result.valid()) {
		SPDLOG_ERROR("function '{}' failed:\n{}", name, result.get<sol::error>().what());
		return iv;
	}

	return result.get<double>();
}

rfkt::anima::call_info_t rfkt::function_table::make_default(std::string_view name) const noexcept {
	const auto& def_args = funcs.find(name)->second.info.args;
	auto ret = std::make_optional<anima::call_info_value_t>();
	ret->name = std::string{ name };
	for (const auto& [arg_name, arg_type] : def_args) {
		ret->args.emplace(arg_name, arg_type.default_value);
	}

	return ret;
}

std::string rfkt::function_table::create_function_source(std::string_view func_name, const func_info& fi) {
	std::string args = {};
	int i = 0;
	for (const auto& [name, _] : fi.args) {
		args += name;
		if (i + 1 != fi.args.size()) args += ", ";
		i++;
	}

	return std::format("function {}(t, iv, {})\n{}\nend", func_name, args, fi.source);
}
