#pragma once

#include <vector_types.h>

namespace rfkt::color {
	auto rgb_to_hsv(const uchar3&)->double3;
	auto hsv_to_rgb(const double3&) -> uchar3;
	auto rgb_to_yuv(const uchar3& rgb) -> double3;
	auto yuv_to_rgb(const double3& yuv) -> uchar3;
}