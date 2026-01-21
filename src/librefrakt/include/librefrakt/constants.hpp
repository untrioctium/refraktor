#pragma once

namespace rfkt {
	constexpr static auto histogram_base_size = 24;
	constexpr static auto histogram_granularity = 4;
	constexpr static auto histogram_size = histogram_base_size * histogram_granularity + 1;
}