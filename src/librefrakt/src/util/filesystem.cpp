#include <filesystem>
#include <fstream>

#ifdef _WIN32
#include <ShlObj.h>
#endif

#include <librefrakt/util/filesystem.h>

using path = rfkt::fs::path;

path& working_directory_global() {
	static path working_directory = std::filesystem::current_path();
	return working_directory;
}

void rfkt::fs::set_working_directory(const path& path) {
	working_directory_global() = path;
	std::filesystem::current_path(path);

}

const path& rfkt::fs::working_directory() {
	return working_directory_global();
}

const path& rfkt::fs::user_local_directory() {
	const static path local_dir = []() -> path {
		path local_path{};
#ifdef _WIN32
		PWSTR base_path = nullptr;
		if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &base_path))) {
			local_path = base_path;
		}
		else {
			local_path = std::filesystem::current_path();
		}
		CoTaskMemFree(base_path);

		local_path /= "refrakt";
#endif
		if (!fs::exists(local_path)) {
			fs::create_directory(local_path);
		}

		return local_path;
	}();

	return local_dir;
}

const path& rfkt::fs::user_home_directory() {
	const static path home_dir = []() -> path {
		#ifdef _WIN32
		PWSTR home_path = nullptr;
		if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Documents, 0, NULL, &home_path))) {
			return home_path;
		}
		else {
			return std::filesystem::current_path();
		}
		#endif
	}();

	return home_dir;
}

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

bool rfkt::fs::remove_directory(const path& dir_path)
{
	return std::filesystem::remove_all(dir_path);
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

bool rfkt::fs::command_in_path(std::string_view command_name)
{
	// get PATH environment variable
	std::string_view path = std::getenv("PATH");

	// split path into individual directories
	std::vector<std::string_view> dirs;
	size_t start = 0;
	size_t end = path.find_first_of(';');

	while (end != std::string::npos) {
		dirs.push_back(path.substr(start, end - start));
		start = end + 1;
		end = path.find_first_of(';', start);
	}

	dirs.push_back(path.substr(start));

	// check if command is in any of the directories
	for (const auto& dir : dirs) {
		fs::path cmd_path = dir;
		cmd_path /= command_name;

		if (std::filesystem::exists(cmd_path)) {
			return true;
		}
	}

	return false;
}
