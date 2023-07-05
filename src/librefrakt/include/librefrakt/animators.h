#pragma once

#include <librefrakt/util/factory.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace rfkt {

	struct animator : factory<animator, const json&> {
		static void init_builtins();

		animator(Key){}

		auto clone() const {
			return make(name(), serialize());
		}

		// need to think of a better way to do this
		// this information is potentially known to the factory
		virtual auto name() const ->std::string = 0;

		virtual auto serialize() const ->json = 0;
		virtual auto apply(double t, double initial) const -> double = 0;
		virtual ~animator() = default;
	};

}