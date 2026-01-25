#pragma once

#include <librefrakt/vector_types.hpp>

namespace rfkt::color {
	auto rgb_to_hsv(const double3&)->double3;
	auto hsv_to_rgb(const double3&) -> double3;
}