#include <algorithm>
#include <librefrakt/util/color.h>
#include <cmath>
// convert a RGB triplet in [0,255] to a hsv triplet in [0,360] for H and [0,1] for SV
auto rfkt::color::rgb_to_hsv(const double3& rgb) -> double3
{
	if (rgb.x == rgb.y && rgb.y == rgb.z) {
		return double3{ 0.0, 0.0, rgb.x };
	}

	auto cmax = std::max(rgb.x, std::max(rgb.y, rgb.z));
	auto cmin = std::min(rgb.x, std::min(rgb.y, rgb.z));
	auto delta = cmax - cmin;

	auto out = double3{ 
		0.0, 
		(cmax == 0.0)? 0: (delta/cmax),
		double(cmax)
	};

	if (cmax == rgb.x)
		out.x = fmod((rgb.y - rgb.z) / delta, 6.0);
	else if (cmax == rgb.y)
		out.x = (rgb.z - rgb.x) / delta + 2.0;
	else if (cmax == rgb.z)
		out.x = (rgb.x - rgb.y) / delta + 4.0;

	out.x *= 60.0;

	if (std::isnan(out.x) || std::isnan(out.y) || std::isnan(out.z)) __debugbreak();

	return out;
}

auto rfkt::color::hsv_to_rgb(const double3& hsv) -> double3 {
	auto c = hsv.z * hsv.y;
	auto x = c * (1.0 - std::abs(std::fmod(hsv.x / 60.0, 2.0) - 1.0));
	auto m = hsv.z - c;

	auto out = double3{ 0.0, 0.0, 0.0 };

	if (hsv.x >= 0.0 && hsv.x < 60.0) {
		out.x = c;
		out.y = x;
		out.z = 0.0;
	}
	else if (hsv.x >= 60.0 && hsv.x < 120.0) {
		out.x = x;
		out.y = c;
		out.z = 0.0;
	}
	else if (hsv.x >= 120.0 && hsv.x < 180.0) {
		out.x = 0.0;
		out.y = c;
		out.z = x;
	}
	else if (hsv.x >= 180.0 && hsv.x < 240.0) {
		out.x = 0.0;
		out.y = x;
		out.z = c;
	}
	else if (hsv.x >= 240.0 && hsv.x < 300.0) {
		out.x = x;
		out.y = 0.0;
		out.z = c;
	}
	else if (hsv.x >= 300.0 && hsv.x < 360.0) {
		out.x = c;
		out.y = 0.0;
		out.z = x;
	}

	out.x += m;
	out.y += m;
	out.z += m;

	return out;
}