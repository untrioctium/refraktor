#pragma once

#include <roccu_vector_types.h>

namespace rfkt::color {
	auto rgb_to_hsv(const double3&)->double3;
	auto hsv_to_rgb(const double3&) -> double3;
}