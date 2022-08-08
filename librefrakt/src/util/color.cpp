#include <algorithm>
#include <cmath>

#include <librefrakt/util/color.h>

// convert a RGB triplet in [0,255] to a hsv triplet in [0,360] for H and [0,1] for SV
auto rfkt::color::rgb_to_hsv(const uchar3& rgb) -> double3
{
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
		out.x = std::fmod((rgb_d.y - rgb_d.z) / delta, 6.0);
	else if (cmax == rgb_d.y)
		out.x = (rgb_d.z - rgb_d.x) / delta + 2.0;
	else if (cmax == rgb_d.z)
		out.x = (rgb_d.x - rgb_d.y) / delta + 4.0;

	out.x *= 60.0;
	return out;
}
