#pragma once

#include <cstddef>
#include <string>
#include <set>
#include <vector>
#include <functional>

namespace rfkt::flame_info {

	namespace def {
		struct common {
			std::string name;
			std::string source;
			std::set<std::string> dependencies;
		};

		struct parameter {
			std::uint32_t index;
			std::string name;
			double default_value = 0.0;
			bool is_precalc;
			std::uint32_t owner;
		};

		struct variation {
			std::uint32_t index;
			std::string name;
			std::string source;
			std::string precalc_source;
			std::vector<std::reference_wrapper<parameter>> parameters;
			std::vector<std::reference_wrapper<common>> dependencies;
			std::set<std::string> flags;
		};
	}

	void initialize(std::string_view config_path);

	auto num_variations()->std::size_t;
	auto num_parameters()->std::size_t;

	bool is_variation(std::string_view name);
	bool is_parameter(std::string_view name);
	bool is_common(std::string_view name);

	auto variation(std::uint32_t idx) -> const def::variation&;
	auto variation(std::string_view name) -> const def::variation&;

	auto parameter(std::uint32_t idx) -> const def::parameter&;
	auto parameter(std::string_view name) -> const def::parameter&;

	auto common(std::string_view name) -> const def::common&;

	auto variations() -> const std::vector<def::variation>&;
}