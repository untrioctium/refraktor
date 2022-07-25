#pragma once

#include <string>
#include <set>
#include <vector>
#include <optional>
#include <map>
#include <vector_types.h>
#include <memory>
#include <algorithm>

#include <librefrakt/traits/hashable.h>
#include <librefrakt/flame_info.h>
#include <librefrakt/animators.h>

namespace rfkt
{
	enum class precision {
		f32,
		f64
	};

	struct animated_double {
		double t0 = 0.0;
		std::unique_ptr<rfkt::animator> ani = nullptr;

		auto sample(double t) const -> double {
			if (!ani) return t0;
			else return ani->apply(t, t0);
		}

		animated_double(const animated_double& o) : t0(o.t0), ani((o.ani) ? o.ani->clone() : nullptr) {}
		auto operator=(const animated_double& o) -> animated_double& {
			t0 = o.t0;
			ani = (o.ani)? o.ani->clone() : nullptr;
			return *this;
		}

		animated_double() = default;
		animated_double(double t0) : t0(t0) {}
	};

	struct affine_matrix {
		animated_double a{1.0}, d{0.0}, b{0.0}, e{1.0}, c{1.0}, f{1.0};

		static affine_matrix identity() {
			return affine_matrix{ 1.0,0.0,0.0,1.0,0.0,0.0 };
		}

		affine_matrix rotate(double deg, double t = 0.0) const noexcept {
			double rad = 0.0174532925199432957692369076848861271344287188854172545609719144 * deg;
			double sino = sin(rad);
			double coso = cos(rad);

			return {
				a.sample(t) * coso + b.sample(t) * sino,
				d.sample(t) * coso + e.sample(t) * sino,
				b.sample(t) * coso - a.sample(t) * sino,
				e.sample(t) * coso - d.sample(t) * sino,
				c.sample(t),
				f.sample(t)
			};
		}
		affine_matrix scale(double scale, double t = 0.0) const noexcept {
			return {
				a.sample(t) * scale,
				d.sample(t) * scale,
				b.sample(t) * scale,
				e.sample(t) * scale,
				c.sample(t),
				f.sample(t)
			};
		}

		affine_matrix translate(double x, double y, double t = 0.0) const noexcept {
			return {
				a.sample(t), d.sample(t), b.sample(t), e.sample(t),
				a.sample(t) * x + b.sample(t) * y + c.sample(t),
				d.sample(t) * x + e.sample(t) * y + f.sample(t)
			};
		}

		double2 apply(const double2& p, double t = 0.0) const noexcept {
			return {
				a.sample(t)* p.x + b.sample(t) * p.y + c.sample(t),
				d.sample(t)* p.x + e.sample(t) * p.y + f.sample(t)
			};
		}

		static affine_matrix from_strings(const std::vector<std::string>& values) {
			affine_matrix r{};
			r.a = std::stod(values[0]);
			r.d = std::stod(values[1]);
			r.b = std::stod(values[2]);
			r.e = std::stod(values[3]);
			r.c = std::stod(values[4]);
			r.f = std::stod(values[5]);
			return r;
		}
	};

	struct vlink: public traits::hashable<vlink> {
		affine_matrix affine = affine_matrix::identity();

		std::pair<animated_double, animated_double> aff_mod_translate = {0.0, 0.0};
		animated_double aff_mod_rotate = { 0.0 };
		animated_double add_mod_scale = { 1.0 };

		static vlink identity() {
			static const auto linear_index = rfkt::flame_info::variation("linear").index;
			auto vl = vlink{};
			vl.variations[linear_index] = 1.0;
			return vl;
		}

		template<typename Callback>
		void for_each_variation(Callback&& cb) const {
			for (const auto& [id, val] : variations)
				cb(flame_info::variation(id), val);
		}

		template<typename Callback>
		void for_each_variation(Callback&& cb) {
			for (auto& [id, val] : variations)
				cb(flame_info::variation(id), val);
		}

		template<typename Callback>
		void for_each_parameter(Callback&& cb) const {
			for (const auto& [id, val] : parameters)
				cb(flame_info::parameter(id), val);
		}

		template<typename Callback>
		void for_each_parameter(Callback&& cb) {
			for (auto& [id, val] : parameters)
				cb(flame_info::parameter(id), val);
		}

		bool has_variation(std::size_t idx) const {
			return variations.contains(idx);
		}

		void add_variation(std::size_t idx, std::optional<double> weight = std::nullopt);
		void remove_variation(std::size_t idx);

		auto real_count() const noexcept -> std::size_t {
			return 6 + variations.size() + parameters.size();
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			auto list = std::vector<std::uint16_t>(variations.size());
			hs.update(std::uint32_t{ 0xFEED });
			for (const auto& [id, val] : variations) list.push_back(id);
			hs.update(list);
		}

	//private:
		std::map<std::size_t, animated_double> variations{};
		std::map<std::size_t, animated_double> parameters{};
	};

	struct xform: public traits::hashable<xform> {
		animated_double weight = 0.0;
		animated_double color = 0.0;
		animated_double color_speed = 0.0;
		animated_double opacity = 0.0;

		std::vector<vlink> vchain;

		static xform identity() {
			auto xf = xform{};
			xf.vchain.emplace_back(vlink::identity());
			return xf;
		}

		auto real_count() const noexcept -> std::size_t {
			std::size_t count = 4;
			for (const auto& vl : vchain) count += vl.real_count();
			return count;
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			hs.update(std::uint32_t{ 0x5EED });
			for (const auto& vl : vchain) vl.add_to_hash(hs);
		}

	};

	class flame: public traits::hashable<flame> {
	public:
		std::vector<xform> xforms = {};
		std::optional<xform> final_xform = std::nullopt;

		std::pair<animated_double, animated_double> center = {0.0, 0.0};
		animated_double scale = 1.0;
		animated_double rotate = 0.0;

		animated_double gamma = 4.0;
		animated_double vibrancy = 1.0;
		animated_double brightness = 1.0;

		flame() = default;
		~flame() = default;

		auto operator=(const flame&) noexcept -> flame&;
		flame(const flame&);

		auto operator=(flame&&) noexcept -> flame& = default;
		flame(flame&&) = default;

		auto real_count() const noexcept -> std::size_t {
			std::size_t count =
				6 + // screen space transform
				1; // weight sum

			if (final_xform.has_value()) count += final_xform->real_count();
			for (const auto& xf : xforms) count += xf.real_count();

			return count;
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			for (const auto& xf : xforms) xf.add_to_hash(hs);

			if (final_xform.has_value()) {
				hs.update(std::uint32_t{ 0xBEEF });
				final_xform->add_to_hash(hs);
			}
		}

		affine_matrix make_screen_space_affine(int w, int h, double t) const noexcept {
			return affine_matrix::identity()
				.translate(w / 2.0, h / 2.0)
				.scale(scale.sample(t) * h)
				.rotate(rotate.sample(t))
				.translate(-center.first.sample(t), -center.second.sample(t));
		}

		affine_matrix make_plane_space_affine(int w, int h, double t) const noexcept {
			return affine_matrix::identity()
				.translate(center.first.sample(t), center.second.sample(t))
				.rotate(-rotate.sample(t))
				.scale(1 / (scale.sample(t) * h))
				.translate(w / -2.0, h / -2.0);
		}

		template<typename Callback>
		void for_each_xform(Callback&& cb) {
			for (int i = 0; i < xforms.size(); i++) {
				cb(i, &xforms[i]);
			}
			if (final_xform.has_value()) cb(-1, &final_xform.value());

			constexpr auto test = sizeof(vlink);
		}

		auto& palette() noexcept { return *palette_hsv; }
		const auto& palette() const noexcept { return *palette_hsv; }

		static auto import_flam3(const std::string& path)->std::optional<flame>;
	private:
		std::unique_ptr<std::array<std::array<animated_double, 3>, 256>> palette_hsv = std::make_unique<std::array<std::array<animated_double, 3>, 256>>();
	};
}