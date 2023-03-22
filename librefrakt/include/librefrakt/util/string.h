#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>

namespace rfkt::str_util {

	auto split(const std::string& str, char delim = ' ') -> std::vector<std::string>;
	auto join(const std::vector<std::string>& tokens, const std::string& delim)->std::string;
	auto join(const std::set<std::string>& tokens, char delim)->std::string;
	auto join(const std::map<std::string, std::string>& pairs, char pair_delim, char item_delim)->std::string;
	auto find_unique(const std::string& str, const std::string& regex)->std::set<std::string>;
}