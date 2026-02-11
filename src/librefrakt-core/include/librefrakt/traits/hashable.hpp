#pragma once

#include <compare>

#include "librefrakt/util/hash.hpp"


namespace rfkt::traits {

	struct hashable {
		auto hash(this const auto& self) -> rfkt::hash_t {
			auto state = rfkt::hash::state_t{};
			self.add_to_hash(state);
			return state.digest();
		}

		constexpr std::strong_ordering operator<=>(const hashable& o) const noexcept = default;
	};

}