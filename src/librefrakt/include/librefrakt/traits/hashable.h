#pragma once

#include <librefrakt/util/hash.h>

namespace rfkt::traits {

	struct hashable {
	public:

		auto hash(this const auto& self) -> rfkt::hash_t {
			auto state = rfkt::hash::state_t{};
			self.add_to_hash(state);
			return state.digest();
		}

	};
}