#include <filesystem>
#include <fstream>

#include <librefrakt/util/filesystem.h>

using path = rfkt::fs::path;

auto rfkt::fs::read_bytes(const path& file_path) -> std::vector<char>
{
	auto file_size = rfkt::fs::size(file_path);
	auto bytes = std::vector<char>{};
	bytes.resize(file_size);

	auto file = std::ifstream{};
	file.open(file_path, std::ios::binary | std::ios::in);
	file.read(bytes.data(), file_size);
	file.close();

	return bytes;
}

auto rfkt::fs::read_string(const path& file_path) -> std::string
{
	std::ifstream f(file_path);
	return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool rfkt::fs::write(const path& file_path, const char* bytes, std::size_t length, bool append)
{
	auto file = std::ofstream{};
	file.open(file_path, std::ios::binary | std::ios::out | (append ? std::ios::app : 0));
	file.write(bytes, length);
	file.close();

	return true;
}

bool rfkt::fs::write(const path& file_path, const std::vector<unsigned char>& bytes, bool append)
{
	return rfkt::fs::write(file_path, (const char* ) bytes.data(), bytes.size(), append);
}

bool rfkt::fs::write(const path& file_path, const std::string& str, bool append)
{
	return rfkt::fs::write(file_path, str.c_str(), str.size(), append);
}

bool rfkt::fs::create_directory(const path& dir_path, bool recursive)
{
	if (recursive)
		return std::filesystem::create_directories(dir_path);
	else
		return std::filesystem::create_directory(dir_path);
}

auto rfkt::fs::list(const path& dir_path, list_filter filter) -> std::vector<path>
{
	if(!rfkt::fs::is_directory(dir_path)) {
		//SPDLOG_WARN("Attemping to list non-directory {}", dir_path.string());
		return {};
	}
	auto listing = std::vector<path>();
	for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
		if (filter(entry.path(), entry.is_directory())) listing.push_back(entry.path());
	}

	return listing;
}

auto rfkt::fs::size(const path& file_path) -> std::size_t
{
	return std::filesystem::file_size(file_path);
}

bool rfkt::fs::exists(const path& path)
{
	return std::filesystem::exists(path);
}

bool rfkt::fs::is_file(const path& path)
{
	return std::filesystem::is_regular_file(path);
}

bool rfkt::fs::is_directory(const path& path)
{
	return std::filesystem::is_directory(path);
}

auto rfkt::fs::last_modified(const path& path) -> long long
{
	if (!rfkt::fs::exists(path)) {
		//SPDLOG_WARN("Cannot stat file: {}", path.string());
		return 0;
	}
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::filesystem::last_write_time(path).time_since_epoch()).count();
}

auto rfkt::fs::now() -> long long
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::filesystem::file_time_type::clock::now().time_since_epoch()).count();
}
