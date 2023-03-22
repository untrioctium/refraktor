#include <sstream>
#include <regex>

#include <librefrakt/util/string.h>

auto rfkt::str_util::split(const std::string& str, char delim) -> std::vector<std::string>
{
	if (str.size() == 0) return {};

    std::vector<std::string> tokens;
    std::string token;
    std::istringstream token_stream{ str };
    while (std::getline(token_stream, token, delim))
    {
        tokens.push_back(token);
    }
    return tokens;
}

auto rfkt::str_util::join(const std::vector<std::string>& tokens, const std::string& delim) -> std::string
{
	if (tokens.empty()) return "";
	auto ret = tokens.at(0);
	for (std::size_t i = 1; i < tokens.size(); i++) ret += delim + tokens.at(i);
	return ret;
}

auto rfkt::str_util::join(const std::set<std::string>& tokens, char delim) -> std::string
{
	if (tokens.size() == 0) return "";
	auto it = tokens.begin();
	auto ret = *it;
	it++;
	for (; it != tokens.end(); it++) ret += delim + *it;
	return ret;
}

auto rfkt::str_util::join(const std::map<std::string, std::string>& pairs, char pair_delim, char item_delim) -> std::string
{
	if (pairs.size() == 0) return "";
	auto it = pairs.begin();
	auto ret = it->first + pair_delim + it->second;
	it++;
	for(; it != pairs.end(); it++) ret += item_delim + it->first + pair_delim + it->second;
	return ret;
}

auto rfkt::str_util::find_unique(const std::string& str, const std::string& regex) -> std::set<std::string>
{
	const auto xcommon_regex = std::regex(regex);

	auto ret = std::set<std::string>();

	for (
		auto i = std::sregex_iterator(str.begin(), str.end(), xcommon_regex);
		i != std::sregex_iterator();
		i++
		)
	{
		if(i->size() > 1) ret.insert(i->str(1));
	}

	return ret;
}
