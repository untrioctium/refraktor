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

		animated_double(double t0, std::unique_ptr<rfkt::animator> ani = nullptr) : t0(t0), ani(std::move(ani)) {}

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

		animated_double(animated_double&& o) noexcept = default;
		animated_double& operator=(animated_double&& o) noexcept = default;

		animated_double() = default;
		~animated_double() = default;

		animated_double make_interpolator(const animated_double& o) const {
			//if (!ani && !o.ani) {
				//if (t0 == o.t0) {
				//	return { t0, nullptr };
				//}

				return {  
					t0,
					animator::make("interpolate", json::object({
						{"smooth", true},
						{"final_value", o.t0}
					})) 
				};
			//}

			/*return {
				t0,
				animator::make("interp_children", json::object({
					{"left_name", (ani)? ani->name(): "noop"},
					{"right_name", (o.ani)? o.ani->name(): "noop"},
					{"left", (ani) ? ani->serialize() : json::object()},
					{"right", (o.ani) ? o.ani->serialize() : json::object()},
					{"right_iv", o.t0}
				}))
			};*/
		}
	};

	struct affine_matrix {
		animated_double a{1.0}, d{0.0}, b{0.0}, e{1.0}, c{0.0}, f{0.0};

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

	struct vlink: public traits::hashable {
		affine_matrix affine = affine_matrix::identity();

		std::pair<animated_double, animated_double> aff_mod_translate = {0.0, 0.0};
		animated_double aff_mod_rotate = { 0.0 };
		animated_double aff_mod_scale = { 1.0 };

		static vlink identity() {
			static const auto linear_index = rfkt::flame_info::variation("linear").index;
			auto vl = vlink{};
			vl.variations[linear_index] = 1.0;
			return vl;
		}

		template<typename Callback>
		void for_each_variation(this auto&& self, Callback&& cb) {
			for (auto& [id, val] : self.variations)
				cb(flame_info::variation(id), val);
		}

		template<typename Callback>
		void for_each_parameter(this auto&& self, Callback&& cb) {
			for (auto& [id, val] : self.parameters)
				cb(flame_info::parameter(id), val);
		}

		template<typename Callback>
		void for_each_precalc(this auto&& self, Callback&& cb) {
			for (auto id : self.precalc)
				cb(flame_info::parameter(id));
		}

		void add_variation(std::uint32_t idx, double weight = 0.0) {
			variations[idx] = weight;

			const auto& vdef = rfkt::flame_info::variation(idx);
			for (const rfkt::flame_info::def::parameter& pdef : vdef.parameters) {
				if (pdef.is_precalc)
					precalc.insert(pdef.index);
				else {
					parameters[pdef.index] = pdef.default_value;
				}
			}
		}

		void remove_variation(std::uint32_t idx) {
			variations.erase(idx);

			const auto& vdef = rfkt::flame_info::variation(idx);
			for (const rfkt::flame_info::def::parameter& pdef : vdef.parameters) {
				if (pdef.is_precalc) 
					precalc.erase(pdef.index);
				else {
					parameters.erase(pdef.index);
				}
			}
		}

		bool has_variation(std::size_t idx) const {
			return variations.contains(idx);
		}

		auto& variation(this auto&& self, std::size_t idx) {
			return self.variations.at(idx);
		}

		auto variation_count() const { return variations.size(); }
		auto precalc_count() const { return precalc.size(); }

		auto& parameter(this auto&& self, std::size_t idx) {
			return self.parameters.at(idx);
		}

		auto real_count() const noexcept -> std::size_t {
			return 6 + variations.size() + parameters.size() + precalc.size();
		}

		void add_to_hash(rfkt::hash::state_t& hs) const {
			auto list = std::vector<std::uint16_t>(variations.size());
			hs.update(std::uint32_t{ 0xFEED });
			for (const auto& [id, val] : variations) list.push_back(id);
			hs.update(list);
		}

		rfkt::animated_double* seek(std::string_view path);
		std::string dump(std::string_view prefix) const;

	private:
		std::map<std::uint32_t, animated_double> variations{};
		std::map<std::uint32_t, animated_double> parameters{};
		std::set<std::uint32_t> precalc{};
	};

	struct xform: public traits::hashable {
		animated_double weight = 0.0;
		animated_double color = 0.0;
		animated_double color_speed = 0.0;
		animated_double opacity = 1.0;

		std::vector<vlink> vchain;

		rfkt::animated_double* seek(std::string_view path);
		std::string dump(std::string_view prefix) const;

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

	class flame: public traits::hashable {
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

		rfkt::animated_double* seek(std::string_view path);
		std::string dump() const;

		auto real_count() const noexcept -> std::size_t {
			std::size_t count =
				6 + // screen space transform
				1; // weight sum

			if (final_xform.has_value()) count += final_xform->real_count();
			for (const auto& xf : xforms) count += xf.real_count();

			return count;
		}

		auto pack(double t, auto&& packer) const {

			auto weight_sum = std::accumulate(xforms.begin(), xforms.end(), 0.0, [](const double& sum, const xform& xf) { return sum + xf.weight.sample(t);  });

			for (int i = 0; i < xforms.size(); i++) {

			}

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
		}

		auto& palette() noexcept { return *palette_hsv; }
		const auto& palette() const noexcept { return *palette_hsv; }

		static auto import_flam3(const std::string& path)->std::optional<flame>;

		void pack(auto&& packer, uint2 dims, double t) const {
			auto mat = make_screen_space_affine(dims.x, dims.y, t);
			packer(mat.a.sample(t));
			packer(mat.d.sample(t));
			packer(mat.b.sample(t));
			packer(mat.e.sample(t));
			packer(mat.c.sample(t));
			packer(mat.f.sample(t));

			auto sum = 0.0;
			for (const auto& xf : xforms) sum += xf.weight.sample(t);
			packer(sum);

			auto pack_xform = [&packer, &t](const xform& xf) {
				packer(xf.weight.sample(t));
				packer(xf.color.sample(t));
				packer(xf.color_speed.sample(t));
				packer(xf.opacity.sample(t));

				for (const auto& vl : xf.vchain) {

					auto affine = vl.affine.scale(vl.aff_mod_scale.sample(t)).rotate(vl.aff_mod_rotate.sample(t)).translate(vl.aff_mod_translate.first.sample(t), vl.aff_mod_translate.second.sample(t));

					packer(affine.a.sample(t));
					packer(affine.d.sample(t));
					packer(affine.b.sample(t));
					packer(affine.e.sample(t));
					packer(affine.c.sample(t));
					packer(affine.f.sample(t));

					vl.for_each_variation([&packer, t](const auto& vdef, const auto& value) { packer(value.sample(t)); });
					vl.for_each_parameter([&packer, t](const auto& pdef, const auto& value) { packer(value.sample(t)); });
					for (int i = 0; i < vl.precalc_count(); i++) packer(-42.42);

				}
			};

			for (const auto& xf : xforms) pack_xform(xf);
			if (final_xform.has_value()) pack_xform(*final_xform);

			for (const auto& hsv : palette()) {
				packer(hsv[0].sample(t));
				packer(hsv[1].sample(t));
				packer(hsv[2].sample(t));
			}
		}

		std::vector<std::uint32_t> make_description() {

			/*const auto xf_size = [](const rfkt::xform& xf) {
				auto ret = std::size_t{ 0 };
				for (const auto& vl : xf.vchain) ret += vl.variations.size();
				return ret + xf.vchain.size();
			};

			std::size_t ret_size = 0;
			for (int i = -1; i < xforms.size(); i++) {
				if (i == -1 && !final_xform.has_value()) continue;
				const auto& xf = (i == -1) ? final_xform.value() : xforms[i];
				ret_size += xf_size(xf) + 1;
			}
			std::vector<std::uint32_t> ret;
			ret.reserve(ret_size);

			const auto make_marker = [](std::uint16_t high, std::uint16_t low) -> std::uint32_t {
				return high << 16 | low;
			};

			for (int i = -1; i < xforms.size(); i++) {
				if (i == -1 && !final_xform.has_value()) continue;
				const auto& xf = (i == -1) ? final_xform.value() : xforms[i];

				auto xfs = xf_size(xf);

				ret.push_back(make_marker((i > 0) ? 0 : 1, xfs));

				for (const auto& vl : xf.vchain) {
					ret.push_back(make_marker(2, vl.variations.size()));

					for (const auto& [id, _] : vl.variations) {
						ret.push_back(id);
					}
				}
			}

			return ret;*/
			return {};

		}
	private:
		std::unique_ptr<std::array<std::array<animated_double, 3>, 256>> palette_hsv = std::make_unique<std::array<std::array<animated_double, 3>, 256>>();
	};
}