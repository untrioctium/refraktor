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
			std::size_t index;
			std::string name;
			double default_value = 0.0;
		};

		struct variation {
			std::size_t index;
			std::string name;
			std::string source;
			std::vector<std::reference_wrapper<parameter>> parameters;
			std::vector<std::reference_wrapper<common>> dependencies;
			std::set<std::string> flags;
		};
	}

	void initialize(const std::string& config_path);

	auto num_variations()->std::size_t;
	auto num_parameters()->std::size_t;

	bool is_variation(const std::string& name);
	bool is_parameter(const std::string& name);
	bool is_common(const std::string& name);

	auto variation(std::size_t idx) -> const def::variation&;
	auto variation(const std::string& name) -> const def::variation&;

	auto parameter(std::size_t idx) -> const def::parameter&;
	auto parameter(const std::string& name) -> const def::parameter&;

	auto common(const std::string& name) -> const def::common&;

	auto variations() -> const std::vector<def::variation>&;
}