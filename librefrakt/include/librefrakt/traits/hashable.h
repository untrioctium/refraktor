#pragma once

#include <librefrakt/util/hash.h>

namespace rfkt::traits {

	struct hashable {
	public:

		template<typename Class>
		auto hash(this const Class& self) -> rfkt::hash_t {
			auto state = rfkt::hash::state_t{};
			self.add_to_hash(state);
			return state.digest();
		}
	};
}