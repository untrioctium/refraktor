#include <variant>

#include <librefrakt/flame_types.h>

namespace rfkt {

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

	double* access(rfkt::flame& flame, const descriptor& desc) noexcept {
		return std::visit([&](auto&& arg) -> double* {
			return arg.access(flame);
		}, desc);
	}

	class anima {
	public:
		enum class target {
			flame,
			xform,
			vlink,
			transform,
			vardata,
			parameter
		};

		flame generate_sample(const flame& base, double t) const noexcept {
			flame result = base;

			for (auto& [desc, args] : animators) {
				auto ptr = access(result, desc);
				if (!ptr) continue;


			}
		}

	private:

		struct func_info {
			std::string name;
			std::map<std::string, std::variant<int, double, bool>> args;
		};

		std::map<descriptor, func_info> animators;
	};
}