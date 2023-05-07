#include <variant>
#include <sol/sol.hpp>
#include <spdlog/spdlog.h>
#include <ranges>

#include <librefrakt/flame_types.h>

namespace rfkt {
	/*
	namespace descriptors {
		struct flame {
			double rfkt::flame::* p;

			constexpr bool operator==(const flame&) const noexcept = default;

			double* access(rfkt::flame& flame) const noexcept {
				return &(flame.*p);
			}
		};

		struct xform {
			int xid;
			double rfkt::xform::* p;

			constexpr bool operator==(const xform&) const noexcept = default;

			double* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				return &(ptr->*p);
			}
		};

		struct vlink {
			int xid;
			int vid;
			double rfkt::vlink::* p;

			constexpr bool operator==(const vlink&) const noexcept = default;

			double* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid <= 0 || vid >= ptr->vchain.size()) return nullptr;
				return &(ptr->vchain[vid].*p);
			}
		};

		struct transform {
			int xid;
			int vid;
			double rfkt::affine::* p;

			constexpr bool operator==(const transform&) const noexcept = default;

			double* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid <= 0 || vid >= ptr->vchain.size()) return nullptr;
				return &(ptr->vchain[vid].transform.*p);
			}
		};

		struct vardata {
			int xid;
			int vid;
			std::string var_name;
			double rfkt::vardata::* p;

			constexpr bool operator==(const vardata&) const noexcept = default;

			double* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid <= 0 || vid >= ptr->vchain.size()) return nullptr;
				if (!ptr->vchain[vid].has_variation(var_name)) return nullptr;
				return &(ptr->vchain[vid][var_name].*p);
			}
		};

		struct parameter {
			int xid;
			int vid;
			std::string var_name;
			std::string param_name;

			constexpr bool operator==(const parameter&) const noexcept = default;

			double* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid <= 0 || vid >= ptr->vchain.size()) return nullptr;
				if (!ptr->vchain[vid].has_variation(var_name)) return nullptr;
				if (!ptr->vchain[vid][var_name].has_parameter(param_name)) return nullptr;
				return &(ptr->vchain[vid][var_name][param_name]);
			};
		};

	}

	using descriptor = std::variant<
		descriptors::flame,
		descriptors::xform,
		descriptors::vlink,
		descriptors::transform,
		descriptors::vardata,
		descriptors::parameter
	>;

	bool operator==(const descriptor& lhs, const descriptor& rhs) noexcept {
		return std::visit([&](auto&& lhs, auto&& rhs) -> bool {
			if constexpr (std::same_as<decltype(lhs), decltype(rhs)>)
				return lhs == rhs;
			else 
				return false;
		}, lhs, rhs);
	}

	double* access(rfkt::flame& flame, const descriptor& desc) noexcept {
		return std::visit([&](auto&& arg) -> double* {
			return arg.access(flame);
		}, desc);
	}
	*/

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