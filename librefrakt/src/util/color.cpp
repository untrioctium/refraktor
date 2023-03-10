#include <algorithm>
#include <librefrakt/util/color.h>
#include <cmath>
// convert a RGB triplet in [0,255] to a hsv triplet in [0,360] for H and [0,1] for SV
auto rfkt::color::rgb_to_hsv(const uchar3& rgb) -> double3
{
	if (rgb.x == rgb.y && rgb.y == rgb.z) {
		return double3{ 0.0, 0.0, rgb.x / 255.0 };
	}
	auto rgb_d = double3{ rgb.x / 255.0, rgb.y / 255.0, rgb.z / 255.0 };
	auto cmax = std::max(rgb_d.x, std::max(rgb_d.y, rgb_d.z));
	auto cmin = std::min(rgb_d.x, std::min(rgb_d.y, rgb_d.z));
	auto delta = cmax - cmin;

	auto out = double3{ 
		0.0, 
		(cmax == 0.0)? 0: (delta/cmax),
		double(cmax)
	};

	if (cmax == rgb_d.x)
		out.x = fmod((rgb_d.y - rgb_d.z) / delta, 6.0);
	else if (cmax == rgb_d.y)
		out.x = (rgb_d.z - rgb_d.x) / delta + 2.0;
	else if (cmax == rgb_d.z)
		out.x = (rgb_d.x - rgb_d.y) / delta + 4.0;

	out.x *= 60.0;

	if (std::isnan(out.x) || std::isnan(out.y) || std::isnan(out.z)) __debugbreak();

	return out;
}

auto rfkt::color::hsv_to_rgb(const double3& hsv) -> uchar3 {
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

	return uchar3{
		unsigned char(out.x * 255.0), 
		unsigned char(out.y * 255.0), 
		unsigned char(out.z * 255.0) 
	};
}

auto rfkt::color::rgb_to_yuv(const uchar3& rgb) -> double3 {
	auto r = rgb.x / 255.0;
	auto g = rgb.y / 255.0;
	auto b = rgb.z / 255.0;

	auto y = 0.299 * r + 0.587 * g + 0.114 * b;
	auto u = -0.14713 * r - 0.28886 * g + 0.436 * b;
	auto v = 0.615 * r - 0.51499 * g - 0.10001 * b;

	return double3{ y, u, v };
}

auto rfkt::color::yuv_to_rgb(const double3& yuv) -> uchar3 {
	auto y = yuv.x;
	auto u = yuv.y;
	auto v = yuv.z;

	auto r = y + 1.13983 * v;
	auto g = y - 0.39465 * u - 0.58060 * v;
	auto b = y + 2.03211 * u;

	return uchar3{
		unsigned char(r * 255.0),
		unsigned char(g * 255.0),
		unsigned char(b * 255.0)
	};
}