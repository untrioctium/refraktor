#pragma once

#include <librefrakt/util/hash.h>

namespace rfkt::traits {
	template<typename Class>
	struct hashable {
	public:
		auto hash() const -> rfkt::hash_t {
			auto state = rfkt::hash::state_t{};
			static_cast<const Class&>(*this).add_to_hash(state);
			return state.digest();
		}
	};
}