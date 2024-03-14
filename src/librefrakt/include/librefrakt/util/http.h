#pragma once

#include <string>
#include <map>

#include <librefrakt/util/filesystem.h>

namespace rfkt::http {

	bool download(std::string_view url, const rfkt::fs::path& path);
	auto download(std::string_view url) -> std::optional<std::vector<char>>;
	auto head(std::string_view url) -> std::optional<std::map<std::string, std::string, std::less<>>>;
}