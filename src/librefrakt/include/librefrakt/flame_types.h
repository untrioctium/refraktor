#pragma once

#include <optional>
#include <map>
#include <vector>
#include <string_view>
#include <variant>

#include <nlohmann/json.hpp>

#include <librefrakt/traits/hashable.h>

#include <experimental/generator>
#include <vector_types.h>

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

namespace rfkt {

	template<typename... Args>
	using generator = std::experimental::generator<Args...>;

	class function_table;
	class flamedb;

	enum class precision { f32, f64 };

	namespace detail {

		template<typename T>
		concept packer = requires(T pack, double v) {
			{ pack(v) };
		};

		template<typename T>
		concept vec2_type = requires(T vec, double v) {
			{vec.x} -> std::same_as<double>;
			{vec.y} -> std::same_as<double>;
			{T{ v, v }} -> std::same_as<T>;
		};

		template<typename T>
		constexpr static const T* typed_nullptr = nullptr;

	}

	struct anima {

		using arg_t = std::variant<int, double, bool, std::string>;
		using arg_map_t = std::map<std::string, arg_t, std::less<>>;
		
		struct call_info_value_t {
			std::string name;
			arg_map_t args;
		};

		using call_info_t = std::optional<call_info_value_t>;

		double t0 = 0.0;
		call_info_t call_info = std::nullopt;

		explicit(false) constexpr anima(double t0) noexcept : t0(t0) {}
		anima(double t0, const call_info_t& args) noexcept : t0(t0), call_info(args) {}
		anima(double t0, call_info_t&& args) noexcept : t0(t0), call_info(std::move(args)) {}

		template<typename Func>
		double sample(double t, Func& invoker) const {
			if (!call_info) return t0;

			return invoker(call_info->name, t, t0, call_info->args);
		}

		anima() = default;

		ordered_json serialize() const noexcept;

		static std::optional<anima> deserialize(const json& js, const function_table& ft) noexcept;

		anima interpolate(anima o, double start_time, double length) const {
			auto new_anima = anima{ t0 };
			new_anima.call_info = call_info_value_t{};
			auto& nargs = new_anima.call_info->args;

			if (call_info) {
				nargs["left.function"] = call_info->name;
				for (const auto& [name, value] : call_info->args) {
					nargs["left." + name] = value;
				}
			}

			nargs["right.t0"] = o.t0;
			if (o.call_info) {
				nargs["right.function"] = o.call_info->name;
				for (const auto& [name, value] : o.call_info->args) {
					nargs["right." + name] = value;
				}
			}

			nargs["start_time"] = start_time;
			nargs["length"] = length;

			return new_anima;
		}
	};

	template<typename Owner>
	using anima_ptr = anima Owner::*;

	struct affine {

		// [a b c]
		// [d e f]

		anima a = 0.0;
		anima d = 0.0;
		anima b = 0.0;
		anima e = 0.0;
		anima c = 0.0;
		anima f = 0.0;

		affine rotated(double deg) const noexcept {
			double rad = -glm::radians(deg);

			glm::dmat4 m = {
				a.t0, b.t0, 0, 0,
				d.t0, e.t0, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
			};

			auto newmat = glm::rotate(m, rad, glm::dvec3(0, 0, 1));

			return {
				newmat[0][0], newmat[1][0], newmat[0][1], newmat[1][1], c.t0, f.t0
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

		affine translated(const detail::vec2_type auto& v) const noexcept {
			return translated(v.x, v.y);
		}

		template<typename T>
		auto pack(T&& p) const noexcept {
			p(a); p(d); p(b); p(e); p(c); p(f);
		}

		rfkt::generator<anima*> pack() {
			co_yield &a; co_yield &d; co_yield &b; co_yield &e; co_yield &c; co_yield &f;
		}

		std::size_t size_reals() const noexcept {
			return 6;
		}

		static auto identity() noexcept {
			return affine{ 1, 0, 0, 1, 0, 0 };
		}

		ordered_json serialize() const noexcept {
			return ordered_json::array({ a.serialize(), d.serialize(), b.serialize(), e.serialize(), c.serialize(), f.serialize() });
		}

		static std::optional<affine> deserialize(const json& js, const function_table& ft) noexcept {
			if (!js.is_array()) return std::nullopt;

			auto arr = js.get<json::array_t>();
			if (arr.size() != 6) return std::nullopt;

			auto a = anima::deserialize(arr[0], ft);
			auto d = anima::deserialize(arr[1], ft);
			auto b = anima::deserialize(arr[2], ft);
			auto e = anima::deserialize(arr[3], ft);
			auto c = anima::deserialize(arr[4], ft);
			auto f = anima::deserialize(arr[5], ft);

			if (!a || !b || !c || !d || !e || !f) return std::nullopt;

			return affine{ std::move(*a), std::move(*d), std::move(*b), std::move(*e), std::move(*c), std::move(*f) };
		}

		template<typename Invoker>
		std::pair<double, double> sample(std::pair<double, double> point, double t, Invoker invoker) {
			return {
				a.sample(t, invoker) * point.first + b.sample(t, invoker) * point.second + c.sample(t, invoker),
				d.sample(t, invoker) * point.first + e.sample(t, invoker) * point.second + f.sample(t, invoker)
			};
		}

		constexpr static std::string_view pointer_to_name(anima_ptr<affine> ptr) {
			if (ptr == &affine::a) return "a";
			if (ptr == &affine::b) return "b";
			if (ptr == &affine::c) return "c";
			if (ptr == &affine::d) return "d";
			if (ptr == &affine::e) return "e";
			if (ptr == &affine::f) return "f";
			return "UNKNOWN";
		}

		constexpr static anima rfkt::affine::* name_to_pointer(std::string_view name) {
			if (name == "a") return &affine::a;
			if (name == "b") return &affine::b;
			if (name == "c") return &affine::c;
			if (name == "d") return &affine::d;
			if (name == "e") return &affine::e;
			if (name == "f") return &affine::f;
			return nullptr;
		}
	};

	class vardata {
	public:
		anima weight;

		auto& operator[](std::string_view name) {
			return parameters_.find(name)->second;
		}

		const auto& operator[](std::string_view name) const {
			return parameters_.find(name)->second;
		}

		auto begin() { return parameters_.begin(); }
		auto end() { return parameters_.end(); }

		auto begin() const { return parameters_.cbegin(); }
		auto end() const { return parameters_.cend(); }

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

		template<typename Invoker>
		rfkt::generator<double> pack_sample(Invoker& i, double t) const {
			co_yield weight.sample(t, i);

			for (const auto& [_, value] : parameters_) {
				co_yield value.sample(t, i);
			}

			for (auto i = 0; i < precalc_count_; i++) {
				co_yield 0.0;
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
			return parameters_.contains(pname);
		}

		vardata() = delete;
		vardata(vardata&&) = default;
		vardata(const vardata&) = default;

		vardata& operator=(vardata&&) = default;
		vardata& operator=(const vardata&) = default;

		~vardata() = default;

		ordered_json serialize() const noexcept {
			if (parameters_.empty()) return weight.serialize();

			ordered_json js;
			js["weight"] = weight.serialize();
			js["parameters"] = json::object();

			for (const auto& [name, value] : parameters_) {
				js["parameters"][name] = value.serialize();
			}

			return js;
		}

		static std::optional<vardata> deserialize(std::string_view name, const json& js, const function_table& ft, const flamedb& fdb);

		constexpr static std::string_view pointer_to_name(anima_ptr<vardata> ptr) {
			if (ptr == &vardata::weight) return "weight";
			return "UNKNOWN";
		}

		constexpr static anima_ptr<vardata> name_to_pointer(std::string_view name) {
			if (name == "weight") return &vardata::weight;
			return nullptr;
		}

		static std::pair<std::string, vardata> identity() {
			return { "linear", vardata{ 1.0, 0, {} } };
		}

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

		template<typename Invoker>
		rfkt::generator<double> pack_sample(Invoker& i, double t) const {
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

			co_yield sampled_aff.a.t0;
			co_yield sampled_aff.d.t0;
			co_yield sampled_aff.b.t0;
			co_yield sampled_aff.e.t0;
			co_yield sampled_aff.c.t0;
			co_yield sampled_aff.f.t0;

			for (const auto& [_, v] : variations_) {
				for (auto ret : v.pack_sample(i, t)) {
					co_yield ret;
				}
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

		ordered_json serialize() const noexcept {
			ordered_json js;
			js["transform"] = transform.serialize();
			js["mod_x"] = mod_x.serialize();
			js["mod_y"] = mod_y.serialize();
			js["mod_scale"] = mod_scale.serialize();
			js["mod_rotate"] = mod_rotate.serialize();
			js["variations"] = ordered_json::object();

			for (const auto& [name, value] : variations_) {
				js["variations"][name] = value.serialize();
			}

			return js;
		}

		static std::optional<vlink> deserialize(const json& js, const function_table& ft, const flamedb& fdb);

		constexpr static std::string_view pointer_to_name(anima_ptr<vlink> ptr) {
			if (ptr == &vlink::mod_x) return "mod_x";
			if (ptr == &vlink::mod_y) return "mod_y";
			if (ptr == &vlink::mod_scale) return "mod_scale";
			if (ptr == &vlink::mod_rotate) return "mod_rotate";
			return "UNKNOWN";
		}

		constexpr static anima_ptr<vlink> name_to_pointer(std::string_view name) {
			if (name == "mod_x") return &vlink::mod_x;
			if (name == "mod_y") return &vlink::mod_y;
			if (name == "mod_scale") return &vlink::mod_scale;
			if (name == "mod_rotate") return &vlink::mod_rotate;
			return nullptr;
		}

		static vlink identity() {
			auto vl = vlink{};
			vl.transform = affine::identity();
			vl.add_variation(vardata::identity());
			return vl;
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

		template<typename Invoker>
		rfkt::generator<double> pack_sample(Invoker& i, double t) const {
			co_yield weight.sample(t, i);
			co_yield color.sample(t, i);
			co_yield color_speed.sample(t, i);
			co_yield opacity.sample(t, i);

			for (const auto& link : vchain) {
				for (auto ret : link.sample(i, t)) {
					co_yield ret;
				}
			}
		}

		auto size_reals() const noexcept {
			auto size = 4;
			for (const auto& vl : vchain) {
				size += vl.size_reals();
			}
			return size;
		}

		ordered_json serialize() const noexcept {
			ordered_json js;
			js["weight"] = weight.serialize();
			js["color"] = color.serialize();
			js["color_speed"] = color_speed.serialize();
			js["opacity"] = opacity.serialize();
			js["vchain"] = ordered_json::array();

			for (const auto& link : vchain) {
				js["vchain"].emplace_back(link.serialize());
			}

			return js;
		}

		static std::optional<xform> deserialize(const json& js, const function_table& ft, const flamedb& fdb);

		constexpr static std::string_view pointer_to_name(anima_ptr<xform> ptr) {
			if (ptr == &xform::weight) return "weight";
			if (ptr == &xform::color) return "color";
			if (ptr == &xform::color_speed) return "color_speed";
			if (ptr == &xform::opacity) return "opacity";
			return "UNKNOWN";
		}

		constexpr static anima_ptr<xform> name_to_pointer(std::string_view name) {
			if (name == "weight") return &xform::weight;
			if (name == "color") return &xform::color;
			if (name == "color_speed") return &xform::color_speed;
			if (name == "opacity") return &xform::opacity;
			return nullptr;
		}

		static xform identity() {
			auto xf = xform{};
			xf.weight = 0.0;
			xf.color = 0.0;
			xf.color_speed = 0.0;
			xf.opacity = 1.0;
			xf.vchain.emplace_back(vlink::identity());
			return xf;
		}

	};

	using palette_t = std::vector<std::array<double, 3>>;


	class flame : public traits::hashable {
	public:

		std::string name;

		std::optional<xform> final_xform;
		std::optional<std::vector<std::vector<anima>>> chaos_table;

		anima center_x;
		anima center_y;
		anima scale;
		anima rotate;

		anima gamma;
		anima brightness;
		anima vibrancy;

		palette_t palette;
		anima mod_hue;
		anima mod_sat;
		anima mod_val;

		flame() = default;

		void add_to_hash(rfkt::hash::state_t& hs) const {
			for (const auto& xf : xforms_) {
				hs.update(0xDULL);
				xf.add_to_hash(hs);
			}

			if (final_xform.has_value()) {
				hs.update(0xFULL);
				final_xform->add_to_hash(hs);
			}

			if (chaos_table.has_value()) {
				hs.update(0xCULL);
			}
		}

		/*template<typename T>
		auto pack(T&& p) const noexcept {
			for (const auto& xf : xforms) {
				xf.pack(std::forward<T>(p));
			}
			if (final_xform) final_xform->pack(std::forward<T>(p));
		}*/

		std::size_t size_reals() const noexcept {
			auto size = final_xform ? final_xform->size_reals() : 0;

			if (chaos_table.has_value()) {
				size += xforms_.size() * (xforms_.size() + 1);
			}

			for (const auto& xf : xforms_) {
				size += xf.size_reals();
			}
			return size + 13;
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

			if (chaos_table.has_value()) {
				for (const auto& row : chaos_table.value()) {
					p(0.0); // space for weight sum
					for (const auto& column : row) {
						p(column.sample(t, i));
					}
				}
			}

			for (const auto& xf : xforms_) {
				xf.pack_sample(p, i, t);
			}
			if (final_xform) final_xform->pack_sample(p, i, t);

			for(const auto& [hue, sat, val]: palette) {
				p(std::fmod(hue + mod_hue.sample(t, i), 360.0)); 
				p(std::clamp(sat + mod_sat.sample(t, i), 0.0, 1.0)); 
				p(std::clamp(val + mod_val.sample(t, i), 0.0, 1.0));
			}
		}

		std::vector<std::size_t> affine_indices() const {

			auto ret = std::vector<std::size_t>{};
			//ret.push_back(0);
			//ret.push_back(6);

			constexpr static std::size_t flame_offset = 13;
			constexpr static std::size_t xform_base_reals = 4;

			std::size_t index = chaos_table.has_value() ? xforms_.size() * (xforms_.size() + 1): 0;
			index += flame_offset;

			for (auto& xf: xforms_) {
				index += xform_base_reals;

				for (auto& vl : xf.vchain) {
					ret.push_back(index);
					index += vl.size_reals();
				}
			}

			if (final_xform) {
				index += xform_base_reals;

				for (auto& vl : final_xform->vchain) {
					ret.push_back(index);
					index += vl.size_reals();
				}
			}

			return ret;
		}

		template<typename Packer, typename Invoker>
		void pack_samples(Packer& p, Invoker& i, double start, double offset, int count, int w, int h) const {
			for (int idx = 0; idx < count; idx++) {
				pack_sample(p, i, start + offset * idx, w, h);
			}
		}

		template<typename T>
		void for_each_xform(this auto&& self, T&& t) noexcept {
			for (int i = 0; i < self.xforms_.size(); i++) {
				t(i, self.xforms_[i]);
			}
			if (self.final_xform) t(-1, *self.final_xform);
		}

		std::span<xform> xforms() noexcept { return xforms_; }
		std::span<const xform> xforms() const noexcept { return xforms_; }

		xform* get_xform(int index) noexcept {
			if (index == -1) return final_xform ? &*final_xform : nullptr;
			if (index < 0 || index >= xforms_.size()) return nullptr;
			return &xforms_[index];
		}

		void clear_xforms() noexcept {
			xforms_.clear();
			chaos_table.reset();
		}

		auto& add_xform(xform&& xf) noexcept {

			if (chaos_table.has_value()) {
				for(auto& row: chaos_table.value()) {
					row.emplace_back(1.0);
				}

				chaos_table->emplace_back();
				for(int i = 0; i < xforms_.size(); i++) {
					chaos_table->back().emplace_back(1.0);
				}
			}

			return xforms_.emplace_back(std::move(xf));
		}

		void add_chaos() noexcept {
			if (chaos_table.has_value()) return;

			chaos_table.emplace();
			for(int i = 0; i < xforms_.size(); i++) {
				chaos_table->emplace_back();
				for(int j = 0; j < xforms_.size(); j++) {
					chaos_table->back().emplace_back(1.0);
				}
			}
		}

		ordered_json serialize() const noexcept;

		static std::optional<flame> deserialize(const json& js, const function_table& ft, const flamedb& fdb);

		constexpr static std::string_view pointer_to_name(anima_ptr<flame> ptr) {
			if (ptr == &flame::center_x) return "center_x";
			if (ptr == &flame::center_y) return "center_y";
			if (ptr == &flame::scale) return "scale";
			if (ptr == &flame::rotate) return "rotate";
			if (ptr == &flame::gamma) return "gamma";
			if (ptr == &flame::brightness) return "brightness";
			if (ptr == &flame::vibrancy) return "vibrancy";
			if (ptr == &flame::mod_hue) return "mod_hue";
			if (ptr == &flame::mod_sat) return "mod_sat";
			if (ptr == &flame::mod_val) return "mod_val";
			return {};
		}

		constexpr static anima_ptr<flame> name_to_pointer(std::string_view name) {
			if(name == "center_x") return &flame::center_x;
			if(name == "center_y") return &flame::center_y;
			if(name == "scale") return &flame::scale;
			if(name == "rotate") return &flame::rotate;
			if(name == "gamma") return &flame::gamma;
			if(name == "brightness") return &flame::brightness;
			if(name == "vibrancy") return &flame::vibrancy;
			if(name == "mod_hue") return &flame::mod_hue;
			if(name == "mod_sat") return &flame::mod_sat;
			if(name == "mod_val") return &flame::mod_val;

			return nullptr;
		}

		rfkt::hash_t value_hash() const noexcept;

	private:

		std::vector<xform> xforms_;

	};

	namespace accessors {
		namespace detail {
			template<typename T>
			constexpr auto cast_ptr(anima_ptr<T> ptr) noexcept {
				return std::bit_cast<std::array<char, sizeof(ptr)>>(ptr);
			}

			template<typename T>
			constexpr auto ptr_name(anima_ptr<T> ptr) noexcept {
				return T::pointer_to_name(ptr);
			}
		}

		struct flame : public traits::hashable {
			anima_ptr<rfkt::flame> p;

			flame(anima_ptr<rfkt::flame> p) noexcept : p(p) { }

			anima* access(rfkt::flame& flame) const noexcept {
				return &(flame.*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(detail::cast_ptr(p));
			}

			std::string to_string() const {
				return std::format("flame.{}", detail::ptr_name(p));
			}

			constexpr std::strong_ordering operator<=>(const flame& o) const noexcept {
				return detail::cast_ptr(p) <=> detail::cast_ptr(o.p);
			}
		};

		struct xform : public traits::hashable {
			int xid;
			anima_ptr<rfkt::xform> p;

			xform(int xid, anima_ptr<rfkt::xform> p) noexcept : xid(xid), p(p) { }

			anima* access(rfkt::flame& flame) const noexcept {
				auto ptr = flame.get_xform(xid);
				if (!ptr) return nullptr;
				return &(ptr->*p);
			}

			void add_to_hash(rfkt::hash::state_t& hs) const {
				hs.update(xid);
				hs.update(detail::cast_ptr(p));
			}

			std::string to_string() const {
				return std::format("flame.xf[{}].{}", xid, detail::ptr_name(p));
			}

			constexpr std::strong_ordering operator<=>(const xform& o) const noexcept {
				if(auto xid_cmp = xid <=> o.xid; xid_cmp != 0) return xid_cmp;
				return detail::cast_ptr(p) <=> detail::cast_ptr(o.p);
			}
		};

		struct vlink : public traits::hashable {
			int xid;
			int vid;
			anima_ptr<rfkt::vlink> p;

			vlink(int xid, int vid, anima_ptr<rfkt::vlink> p) noexcept : xid(xid), vid(vid), p(p) { }

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

			std::string to_string() const {
				return std::format("flame.xf[{}].vl[{}].{}", xid, vid, detail::ptr_name(p));
			}

			constexpr std::strong_ordering operator<=>(const vlink& o) const noexcept {
				if(auto xid_cmp = xid <=> o.xid; xid_cmp != 0) return xid_cmp;
				if(auto vid_cmp = vid <=> o.vid; vid_cmp != 0) return vid_cmp;
				return detail::cast_ptr(p) <=> detail::cast_ptr(o.p);
			}
		};

		struct transform : public traits::hashable {
			int xid;
			int vid;
			anima_ptr<rfkt::affine> p;

			transform(int xid, int vid, anima_ptr<rfkt::affine> p) noexcept : xid(xid), vid(vid), p(p) { }
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

			std::string to_string() const {
				return std::format("flame.xf[{}].vl[{}].transform.{}", xid, vid, detail::ptr_name(p));
			}

			constexpr std::strong_ordering operator<=>(const transform& o) const noexcept {
				if(auto xid_cmp = xid <=> o.xid; xid_cmp != 0) return xid_cmp;
				if(auto vid_cmp = vid <=> o.vid; vid_cmp != 0) return vid_cmp;
				return detail::cast_ptr(p) <=> detail::cast_ptr(o.p);
			}
		};

		struct vardata : public traits::hashable {
			int xid;
			int vid;
			std::string var_name;
			anima_ptr<rfkt::vardata> p;

			vardata(int xid, int vid, std::string var_name, anima_ptr<rfkt::vardata> p) noexcept : xid(xid), vid(vid), var_name(std::move(var_name)), p(p) { }

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

			std::string to_string() const {
				return std::format("flame.xf[{}].vl[{}].var[{}].{}", xid, vid, var_name, detail::ptr_name(p));
			}

			constexpr std::strong_ordering operator<=>(const vardata& o) const noexcept {
				if(auto xid_cmp = xid <=> o.xid; xid_cmp != 0) return xid_cmp;
				if(auto vid_cmp = vid <=> o.vid; vid_cmp != 0) return vid_cmp;
				if(auto var_name_cmp = var_name <=> o.var_name; var_name_cmp != 0) return var_name_cmp;
				return detail::cast_ptr(p) <=> detail::cast_ptr(o.p);
			}
		};

		struct parameter : public traits::hashable {
			int xid;
			int vid;
			std::string var_name;
			std::string param_name;

			parameter(int xid, int vid, std::string var_name, std::string param_name) noexcept : xid(xid), vid(vid), var_name(std::move(var_name)), param_name(std::move(param_name)) { }

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

			std::string to_string() const {
				return std::format("flame.xf[{}].vl[{}].var[{}].{}", xid, vid, var_name, param_name);
			}

			constexpr std::strong_ordering operator<=>(const parameter& o) const noexcept {
				if(auto xid_cmp = xid <=> o.xid; xid_cmp != 0) return xid_cmp;
				if(auto vid_cmp = vid <=> o.vid; vid_cmp != 0) return vid_cmp;
				if(auto var_name_cmp = var_name <=> o.var_name; var_name_cmp != 0) return var_name_cmp;
				return param_name <=> o.param_name;
			}
		};

	}

	using accessor_base = std::variant<
		accessors::flame,
		accessors::xform,
		accessors::vlink,
		accessors::transform,
		accessors::vardata,
		accessors::parameter
	>;

	struct accessor : public accessor_base, public traits::hashable {
		using accessor_base::accessor_base;

		anima* access(rfkt::flame& flame) const noexcept {
			return std::visit([&flame](const auto& arg) -> anima* {
				return arg.access(flame);
				}, *this);
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			std::visit([&hs](const auto& arg) {
				arg.add_to_hash(hs);
			}, *this);
		}

		std::string to_string() const {
			return std::visit([](const auto& arg) -> std::string {
				return arg.to_string();
			}, *this);
		}

		constexpr std::strong_ordering operator<=>(const accessor& o) const noexcept {

			if(auto tag_cmp = index() <=> o.index(); tag_cmp != std::strong_ordering::equivalent)
				return tag_cmp;

			return std::visit(
				[this]<typename RightT>(const RightT& rhs) -> std::strong_ordering {
					return std::get<RightT>(*this) <=> rhs;
				}, o);
		}
	};

	class flamedb;

	auto import_flam3(const flamedb&, std::string_view content) noexcept -> std::optional<flame>;
}