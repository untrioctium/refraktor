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

	constexpr std::pair<std::string_view, std::string_view> chomp(std::string_view path, std::string_view delim) noexcept {
		if (!path.contains(delim)) return { path, {} };

		const auto delim_pos = path.find(delim);
		const auto delim_length = delim.size();

		auto left = path.substr(0, delim_pos);
		auto right = path.substr(delim_pos + delim_length);

		return { left, right };
	}

	static_assert(chomp("hello/world", "/").first == "hello");
	static_assert(chomp("hello/world", "/").second == "world");
	static_assert(chomp("hello", "/").first == "hello");
	static_assert(chomp("hello", "/").second.size() == 0);

	struct affine {
		double a = 0.0;
		double d = 0.0;
		double b = 0.0;
		double e = 0.0;
		double c = 0.0;
		double f = 0.0;

		affine rotated(double degrees) const noexcept;

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

		auto pack(packer auto p) const {
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
			return nullptr;
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

		auto pack(packer auto p) const {
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
			if (auto iter = self.parameters_.find(root); iter != self.paramters_.end()) {
				return &iter->second;
			}
			return nullptr;
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

		auto pack(packer auto p) const {
			transform.pack(p);
			for (const auto& [_, v] : variations_) {
				v.pack(p);
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

			return nullptr;
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

		auto pack(packer auto&& p) const {
			p(weight); p(color); p(color_speed); p(opacity);

			for (const auto& link : vchain) {
				link.pack(p);
			}
		}

		auto size_reals() const noexcept {
			auto size = 4;
			for (const auto& vl : vchain) {
				size += vl.size_reals();
			}
			return size;
		}

		auto seek(this auto&& self, std::string_view path) {
			auto [root, stem] = chomp(path, "/");
			if (root == "weight") return &self.weight;
			if (root == "color") return &self.color;
			if (root == "color_speed") return &self.color_speed;
			if (root == "opacity") return &self.opacity;
		}
	};

	using palette = std::vector<std::array<double, 3>>;

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

		void pack(packer auto&& p) const noexcept {
			for (const auto& xf : xforms) {
				xf.pack(p);
			}
			if (final_xform) final_xform->pack(p);
		}

		std::size_t size_reals() const noexcept {
			auto size = (final_xform) ? final_xform->size_reals() : 0;
			for (const auto& xf : xforms) {
				size += xf.size_reals();
			}
			return size;
		}
	};

	struct flame_import_result {
		flame f;
		palette p;
	};

	class flamedb;

	auto import_flam3(const flamedb&, std::string_view content) noexcept -> std::optional<flame_import_result>;
}