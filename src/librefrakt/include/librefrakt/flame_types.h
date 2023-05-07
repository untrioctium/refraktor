#pragma once

#include <optional>
#include <map>
#include <vector>
#include <string_view>
#include <variant>

#include <librefrakt/traits/hashable.h>

namespace rfkt {

	enum class precision { f32, f64 };

	template<typename T>
	concept packer = requires(T pack, double v) {
		{ pack(v) };
	};

	template<typename T>
	concept vec2_type = requires(T vec, double v) {
		{vec.x} -> std::same_as<double>;
		{vec.y} -> std::same_as<double>;
		{T{v, v}} -> std::same_as<T>;
	};

	template<typename T>
	concept seekable = requires(T obj, std::string_view path) {
		{obj.seek(path)} -> std::same_as<double*>;
	};

	template<typename T>
	constexpr static const T* typed_nullptr = nullptr;

	struct anima {

		using arg_t = std::variant<int, double, bool>;
		using arg_map_t = std::map<std::string, arg_t, std::less<>>;
		using call_info_t = std::optional<std::pair<std::string, arg_map_t>>;

		double t0 = 0.0;
		call_info_t call_info = std::nullopt;

		explicit(false) constexpr anima(double t0) noexcept : t0(t0) {}
		anima(double t0, const call_info_t& args) noexcept : t0(t0), call_info(args) {}

		template<typename Func>
		double sample(double t, Func& invoker) const {
			if (!call_info) return t0;

			return invoker(call_info->first, t, t0, call_info->second);
		}

		anima() = default;
	};


	struct affine {
		anima a = 0.0;
		anima d = 0.0;
		anima b = 0.0;
		anima e = 0.0;
		anima c = 0.0;
		anima f = 0.0;

		affine rotated(double deg) const noexcept {
			double rad = 0.0174532925199432957692369076848861271344287188854172545609719144 * deg;
			double sino = sin(rad);
			double coso = cos(rad);

			return {
				a.t0 * coso + b.t0 * sino,
				d.t0 * coso + e.t0 * sino,
				b.t0 * coso - a.t0 * sino,
				e.t0 * coso - d.t0 * sino,
				c,
				f
			};
		}

		affine scaled(double scale) const noexcept {
			return { a.t0 * scale, d.t0 * scale, b.t0 * scale, e.t0 * scale, c.t0, f.t0 };
		}

		affine translated(double x, double y) const noexcept {
			return {
				a.t0, d.t0, b.t0, e.t0,
				a.t0 * x + b.t0 * y + c.t0,
				d.t0 * x + e.t0 * y + f.t0
			};
		}

		affine translated(const vec2_type auto& v) const noexcept {
			return translated(v.x, v.y);
		}

		template<typename T>
		auto pack(T&& p) const noexcept {
			p(a); p(d); p(b); p(e); p(c); p(f);
		}

		std::size_t size_reals() const noexcept {
			return 6;
		}

		static auto identity() noexcept {
			return affine{ 1, 0, 0, 1, 0, 0 };
		}

	};

	class flamedb;
	class vardata {
	public:
		anima weight;

		auto& operator[](this auto&& self, std::string_view name) {
			return self.parameters_.find(name)->second;
		}

		auto begin(this auto&& self) {
			return self.parameters_.begin();
		}

		auto end(this auto&& self) {
			return self.parameters_.end();
		}

		template<typename T>
		auto pack(T&& p) const noexcept {
			p(weight);
			for (const auto& [_, value] : parameters_) {
				p(value);
			}

			for (auto i = 0; i < precalc_count_; i++) {
				p(0.0);
			}
		}

		template<typename Packer, typename Invoker>
		auto pack_sample(Packer& p, Invoker& i, double t) const {
			p(weight.sample(t, i));

			for (const auto& [_, value] : parameters_) {
				p(value.sample(t, i));
			}

			for (auto i = 0; i < precalc_count_; i++) {
				p(0.0);
			}
		}

		auto size_reals() const noexcept {
			return parameters_.size() + precalc_count_ + 1;
		}

		bool has_parameter(std::string_view pname) const noexcept {
			return parameters_.find(pname) != parameters_.end();
		}

		vardata() = delete;
		vardata(vardata&&) = default;
		vardata(const vardata&) = default;

		vardata& operator=(vardata&&) = default;
		vardata& operator=(const vardata&) = default;

		~vardata() = default;

	private:

		friend class flamedb;
		vardata(double weight, std::size_t precalc_count, std::map<std::string, anima, std::less<>>&& parameters)
			: weight(weight), precalc_count_(precalc_count), parameters_(std::move(parameters)) { }

		std::map<std::string, anima, std::less<>> parameters_;
		std::size_t precalc_count_;
	};

	class vlink : public traits::hashable {
	public:
		affine transform;

		anima mod_x = 0;
		anima mod_y = 0;
		anima mod_scale = 1;
		anima mod_rotate = 0;

		auto& operator[](this auto&& self, std::string_view name) {
			return self.variations_.find(name)->second;
		}

		auto begin(this auto&& self) {
			return self.variations_.begin();
		}

		auto end(this auto&& self) {
			return self.variations_.end();
		}

		void add_variation(std::pair<std::string, vardata>&& vdata) {
			variations_.insert_or_assign(std::move(vdata.first), std::move(vdata.second));
		}

		void remove_variation(std::string_view name) {
			variations_.erase(name);
		}

		bool has_variation(std::string_view name) const {
			return variations_.contains(name);
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			for (const auto& [name, _] : variations_) {
				hs.update(name);
			}
		}

		template<typename T>
		auto pack(T&& p) const noexcept {
			transform.pack(std::forward<T>(p));
			for (const auto& [_, v] : variations_) {
				v.pack(std::forward<T>(p));
			}
		}

		template<typename Packer, typename Invoker>
		void pack_sample(Packer& p, Invoker& i, double t) const {
			
			auto sampled_aff = 
				rfkt::affine{
					transform.a.sample(t, i),
					transform.d.sample(t, i),
					transform.b.sample(t, i),
					transform.e.sample(t, i),
					transform.c.sample(t, i),
					transform.f.sample(t, i),
				}
				.scaled(mod_scale.sample(t, i))
				.rotated(mod_rotate.sample(t, i))
				.translated(mod_x.sample(t, i), mod_y.sample(t, i));

			p(sampled_aff.a.t0);
			p(sampled_aff.d.t0);
			p(sampled_aff.b.t0);
			p(sampled_aff.e.t0);
			p(sampled_aff.c.t0);
			p(sampled_aff.f.t0);

			for (const auto& [_, v] : variations_) {
				v.pack_sample(p, i, t);
			}
		}

		auto size_variations() const noexcept {
			return variations_.size();
		}

		auto size_reals() const noexcept {
			auto size = transform.size_reals();

			for (const auto& [_, v] : variations_) {
				size += v.size_reals();
			}
			return size;
		}

	private:
		std::map<std::string, vardata, std::less<>> variations_;
	};

	struct xform : public traits::hashable {
		anima weight = 0.0;
		anima color = 0.0;
		anima color_speed = 0.0;
		anima opacity = 0.0;

		std::vector<vlink> vchain;

		void add_to_hash(rfkt::hash::state_t& hs) const {
			for (int i = 0; i < vchain.size(); i++) {
				hs.update(0xBULL);
				vchain[i].add_to_hash(hs);
			}
		}

		template<typename T>
		auto pack(T&& p) const noexcept {
			p(weight); p(color); p(color_speed); p(opacity);

			for (const auto& link : vchain) {
				link.pack(std::forward<T>(p));
			}
		}

		template<typename Packer, typename Invoker>
		void pack_sample(Packer& p, Invoker& i, double t) const {
			p(weight.sample(t, i));
			p(color.sample(t, i));
			p(color_speed.sample(t, i));
			p(opacity.sample(t, i));

			for (const auto& link : vchain) {
				link.pack_sample(p, i, t);
			}
		}

		auto size_reals() const noexcept {
			auto size = 4;
			for (const auto& vl : vchain) {
				size += vl.size_reals();
			}
			return size;
		}

	};

	using palette_t = std::vector<std::array<double, 3>>;

	class flame : public traits::hashable {
	public:
		std::vector<xform> xforms;
		std::optional<xform> final_xform;

		anima center_x;
		anima center_y;
		anima scale;
		anima rotate;

		anima gamma;
		anima brightness;
		anima vibrancy;

		palette_t palette;

		flame() = default;

		void add_to_hash(rfkt::hash::state_t& hs) const {
			for (const auto& xf : xforms) {
				hs.update(0xDULL);
				xf.add_to_hash(hs);
			}

			if (final_xform.has_value()) {
				hs.update(0xFULL);
				final_xform->add_to_hash(hs);
			}
		}

		template<typename T>
		auto pack(T&& p) const noexcept {
			for (const auto& xf : xforms) {
				xf.pack(std::forward<T>(p));
			}
			if (final_xform) final_xform->pack(std::forward<T>(p));
		}

		std::size_t size_reals() const noexcept {
			auto size = (final_xform) ? final_xform->size_reals() : 0;
			for (const auto& xf : xforms) {
				size += xf.size_reals();
			}
			return size;
		}

		template<typename Func>
		affine make_screen_space_affine(int w, int h, double t, Func& invoker) const noexcept {
			return affine::identity()
				.translated(w / 2.0, h / 2.0)
				.scaled(scale.sample(t, invoker) * h)
				.rotated(rotate.sample(t, invoker))
				.translated(-center_x.sample(t, invoker), -center_y.sample(t, invoker));
		}

		template<typename Func>
		affine make_plane_space_affine(int w, int h, double t, Func& invoker) const noexcept {
			return affine::identity()
				.translated(center_x.sample(t, invoker), center_y.sample(t, invoker))
				.rotated(-rotate.sample(t, invoker))
				.scaled(1 / (scale.sample(t, invoker) * h))
				.translated(w / -2.0, h / -2.0);
		}

		template<typename Packer, typename Invoker>
		void pack_sample(Packer& p, Invoker& i, double t, int w, int h) const {

			auto screen_space = make_screen_space_affine(w, h, t, i);
			auto plane_space = make_plane_space_affine(w, h, t, i);

			p(screen_space.a.t0);
			p(screen_space.d.t0);
			p(screen_space.b.t0);
			p(screen_space.e.t0);
			p(screen_space.c.t0);
			p(screen_space.f.t0);
			p(plane_space.a.t0);
			p(plane_space.d.t0);
			p(plane_space.b.t0);
			p(plane_space.e.t0);
			p(plane_space.c.t0);
			p(plane_space.f.t0);
			p(0.0); // space for weight sum

			for (const auto& xf : xforms) {
				xf.pack_sample(p, i, t);
			}
			if (final_xform) final_xform->pack_sample(p, i, t);

			for(const auto& [hue, sat, val]: palette) {
				p(hue); p(sat); p(val);
			}
		}

		template<typename T>
		void for_each_xform(this auto&& self, T&& t) noexcept {
			for (int i = 0; i < self.xforms.size(); i++) {
				t(i, self.xforms[i]);
			}
			if (self.final_xform) t(-1, *self.final_xform);
		}

		xform* get_xform(int index) noexcept {
			if (index == -1) return final_xform ? &*final_xform : nullptr;
			if (index < 0 || index >= xforms.size()) return nullptr;
			return &xforms[index];
		}
	};

	namespace descriptors {
		namespace detail {
			template<typename T, typename Class>
			auto cast_ptr(T Class::* ptr) noexcept {
				return std::bit_cast<std::array<char, sizeof(ptr)>>(ptr);
			}
		}

		struct flame : public traits::hashable {
			anima rfkt::flame::* p;

			flame(anima rfkt::flame::* p) noexcept : p(p) { }

			constexpr bool operator==(const flame&) const noexcept = default;

			anima* access(rfkt::flame& flame) const noexcept {
				return &(flame.*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(detail::cast_ptr(p));
			}
		};

		struct xform : public traits::hashable {
			int xid;
			anima rfkt::xform::* p;

			xform(int xid, anima rfkt::xform::* p) noexcept : xid(xid), p(p) { }
			constexpr bool operator==(const xform&) const noexcept = default;

			anima* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				return &(ptr->*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(xid);
				hs.update(detail::cast_ptr(p));
			}
		};

		struct vlink : public traits::hashable {
			int xid;
			int vid;
			anima rfkt::vlink::* p;

			vlink(int xid, int vid, anima rfkt::vlink::* p) noexcept : xid(xid), vid(vid), p(p) { }
			constexpr bool operator==(const vlink&) const noexcept = default;

			anima* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid < 0 || vid >= ptr->vchain.size()) return nullptr;
				return &(ptr->vchain[vid].*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(xid);
				hs.update(vid);
				hs.update(detail::cast_ptr(p));
			}
		};

		struct transform : public traits::hashable {
			int xid;
			int vid;
			anima rfkt::affine::* p;

			transform(int xid, int vid, anima rfkt::affine::* p) noexcept : xid(xid), vid(vid), p(p) { }
			constexpr bool operator==(const transform&) const noexcept = default;

			anima* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid < 0 || vid >= ptr->vchain.size()) return nullptr;
				return &(ptr->vchain[vid].transform.*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(xid);
				hs.update(vid);
				hs.update(detail::cast_ptr(p));
			}
		};

		struct vardata : public traits::hashable {
			int xid;
			int vid;
			std::string var_name;
			anima rfkt::vardata::* p;

			vardata(int xid, int vid, std::string var_name, anima rfkt::vardata::* p) noexcept : xid(xid), vid(vid), var_name(std::move(var_name)), p(p) { }
			constexpr bool operator==(const vardata&) const noexcept = default;

			anima* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid < 0 || vid >= ptr->vchain.size()) return nullptr;
				if (!ptr->vchain[vid].has_variation(var_name)) return nullptr;
				return &(ptr->vchain[vid][var_name].*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(xid);
				hs.update(vid);
				hs.update(var_name);
				hs.update(detail::cast_ptr(p));
			}
		};

		struct parameter : public traits::hashable {
			int xid;
			int vid;
			std::string var_name;
			std::string param_name;

			parameter(int xid, int vid, std::string var_name, std::string param_name) noexcept : xid(xid), vid(vid), var_name(std::move(var_name)), param_name(std::move(param_name)) { }
			constexpr bool operator==(const parameter&) const noexcept = default;

			anima* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				if (vid < 0 || vid >= ptr->vchain.size()) return nullptr;
				if (!ptr->vchain[vid].has_variation(var_name)) return nullptr;
				if (!ptr->vchain[vid][var_name].has_parameter(param_name)) return nullptr;
				return &(ptr->vchain[vid][var_name][param_name]);
			};

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(xid);
				hs.update(vid);
				hs.update(var_name);
				hs.update(param_name);
			}
		};

	}

	using descriptor_base = std::variant<
		descriptors::flame,
		descriptors::xform,
		descriptors::vlink,
		descriptors::transform,
		descriptors::vardata,
		descriptors::parameter
	>;

	struct descriptor : public descriptor_base, public traits::hashable {
		using descriptor_base::descriptor_base;

		anima* access(rfkt::flame& flame) const noexcept {
			return std::visit([&](auto&& arg) -> anima* {
				return arg.access(flame);
				}, *static_cast<const descriptor_base*>(this));
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			std::visit([&](auto&& arg) {
				arg.add_to_hash(hs);
			}, *static_cast<const descriptor_base*>(this));
		}
	};

	/*inline bool operator==(const descriptor& lhs, const descriptor& rhs) noexcept {
		return std::visit([&](auto&& lhs, auto&& rhs) -> bool {
			if constexpr (std::same_as<decltype(lhs), decltype(rhs)>)
			return lhs == rhs;
			else
				return false;
			}, lhs, rhs);
	}*/

	class flamedb;

	auto import_flam3(const flamedb&, std::string_view content) noexcept -> std::optional<flame>;
}