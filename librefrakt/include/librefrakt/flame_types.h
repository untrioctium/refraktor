#pragma once

#include <optional>
#include <map>
#include <vector>
#include <string_view>

#include <nlohmann/json.hpp>

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
	constexpr static T* typed_nullptr = nullptr;

	constexpr std::pair<std::string_view, std::string_view> chomp(std::string_view path, std::string_view delim) noexcept {
		if (!path.contains(delim)) return { path, {} };

		const auto delim_pos = path.find(delim);
		const auto delim_length = delim.size();

		auto left = path.substr(0, delim_pos);
		auto right = path.substr(delim_pos + delim_length);

		return { left, right };
	}

	constexpr std::optional<std::size_t> string_to_sizet(std::string_view v) noexcept {
		if (v.empty()) return std::nullopt;

		std::size_t result = 0;
		for (const auto c : v) {
			if (c < '0' || c > '9') return std::nullopt;
			result *= 10;
			result += c - '0';
		}

		return result;
	}

	static_assert(chomp("hello/world", "/").first == "hello");
	static_assert(chomp("hello/world", "/").second == "world");
	static_assert(chomp("hello", "/").first == "hello");
	static_assert(chomp("hello", "/").second.empty());

	struct affine {
		double a = 0.0;
		double d = 0.0;
		double b = 0.0;
		double e = 0.0;
		double c = 0.0;
		double f = 0.0;

		affine rotated(double deg) const noexcept {
			double rad = 0.0174532925199432957692369076848861271344287188854172545609719144 * deg;
			double sino = sin(rad);
			double coso = cos(rad);

			return {
				a * coso + b * sino,
				d * coso + e * sino,
				b * coso - a * sino,
				e * coso - d * sino,
				c,
				f
			};
		}

		constexpr affine scaled(double scale) const noexcept {
			return { a * scale, d * scale, b * scale, e * scale, c, f };
		}

		constexpr affine translated(double x, double y) const noexcept {
			return {
				a, d, b, e,
				a * x + b * y + c,
				d * x + e * y + f
			};
		}

		constexpr affine translated(const vec2_type auto& v) const noexcept {
			return translated(v.x, v.y);
		}

		constexpr auto apply(const vec2_type auto& v) const noexcept {
			return decltype(v){ a * v.x + b * v.y + c, d * v.x + e * v.y + f };
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

		auto seek(this auto&& self, std::string_view path) noexcept {
			if (path == "a") return &self.a;
			if (path == "b") return &self.b;
			if (path == "c") return &self.c;
			if (path == "d") return &self.d;
			if (path == "e") return &self.e;
			if (path == "f") return &self.f;
			return typed_nullptr<double>;
		}
	};

	class flamedb;
	class vardata {
	public:
		double weight;

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

		auto size_reals() const noexcept {
			return parameters_.size() + precalc_count_ + 1;
		}

		auto seek(this auto&& self, std::string_view path) noexcept {
			auto [root, stem] = chomp(path, "/");
			if (path == "weight") return &self.weight;
			if (auto iter = self.parameters_.find(root); iter != self.parameters_.end()) {
				return &iter->second;
			}
			return typed_nullptr<double>;
		}

		vardata() = delete;
		vardata(vardata&&) = default;
		vardata(const vardata&) = default;

		vardata& operator=(vardata&&) = default;
		vardata& operator=(const vardata&) = default;

		~vardata() = default;

	private:

		friend class flamedb;
		vardata(double weight, std::size_t precalc_count, std::map<std::string, double, std::less<>>&& parameters)
			: weight(weight), precalc_count_(precalc_count), parameters_(std::move(parameters)) { }

		std::map<std::string, double, std::less<>> parameters_;
		std::size_t precalc_count_;
	};

	class vlink : public traits::hashable {
	public:
		affine transform;
		double per_loop = 0;

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

		auto seek(this auto&& self, std::string_view path) noexcept {
			auto [root, stem] = chomp(path, "/");
			if (root == "transform") return self.transform.seek(stem);
			if (auto iter = self.variations_.find(root); iter != self.variations_.end()) {
				return iter->second.seek(stem);
			}

			return typed_nullptr<double>;
		}

	private:
		std::map<std::string, vardata, std::less<>> variations_;
	};

	struct xform : public traits::hashable {
		double weight = 0.0;
		double color = 0.0;
		double color_speed = 0.0;
		double opacity = 0.0;

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

		auto size_reals() const noexcept {
			auto size = 4;
			for (const auto& vl : vchain) {
				size += vl.size_reals();
			}
			return size;
		}

		auto seek(this auto&& self, std::string_view path) noexcept {
			auto [root, stem] = chomp(path, "/");
			if (root == "weight") return &self.weight;
			if (root == "color") return &self.color;
			if (root == "color_speed") return &self.color_speed;
			if (root == "opacity") return &self.opacity;

			if (root == "vlink") {
				auto [vroot, vstem] = chomp(stem, "/");
				auto index = string_to_sizet(vroot);
				if(!index || index.value() >= self.vchain.size()) return typed_nullptr<double>;
				return self.vchain[*index].seek(vstem);
			}

			return typed_nullptr<double>;
		}
	};

	using palette_t = std::vector<std::array<double, 3>>;

	class flame : public traits::hashable {
	public:
		std::vector<xform> xforms;
		std::optional<xform> final_xform;

		double center_x;
		double center_y;
		double scale;
		double rotate;

		double gamma;
		double brightness;
		double vibrancy;

		palette_t palette;

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

		auto seek(this auto&& self, std::string_view path) noexcept {
			auto [root, stem] = chomp(path, "/");
			if (root == "center_x") return &self.center_x;
			if (root == "center_y") return &self.center_y;
			if (root == "scale") return &self.scale;
			if (root == "rotate") return &self.rotate;
			if (root == "gamma") return &self.gamma;
			if (root == "brightness") return &self.brightness;
			if (root == "vibrancy") return &self.vibrancy;

			if (root == "xform") {
				auto [vroot, vstem] = chomp(stem, "/");
				if (vroot == "final") {
					if (!self.final_xform) return typed_nullptr<double>;
					return self.final_xform->seek(vstem); 
				}

				auto index = string_to_sizet(vroot);
				if (!index || index.value() >= self.xforms.size()) {
					return typed_nullptr<double>;
				}

				return self.xforms[*index].seek(vstem);
			}

			return typed_nullptr<double>;
		}

		affine make_screen_space_affine(int w, int h) const noexcept {
			return affine::identity()
				.translated(w / 2.0, h / 2.0)
				.scaled(scale * h)
				.rotated(rotate)
				.translated(-center_x, -center_y);
		}

		affine make_plane_space_affine(int w, int h) const noexcept {
			return affine::identity()
				.translated(center_x, center_y)
				.rotated(-rotate)
				.scaled(1 / (scale * h))
				.translated(w / -2.0, h / -2.0);
		}

		template<typename T>
		void for_each_xform(T&& t) noexcept {
			for (int i = 0; i < xforms.size(); i++) {
				t(i, xforms[i]);
			}
			if (final_xform) t(-1, *final_xform);
		}
	};

	class flamedb;

	auto import_flam3(const flamedb&, std::string_view content) noexcept -> std::optional<flame>;
}