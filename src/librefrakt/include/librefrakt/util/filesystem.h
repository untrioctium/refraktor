#pragma once
#include <filesystem>
#include <functional>

namespace rfkt::fs {

	using path = std::filesystem::path;

	void set_working_directory(const path& path);
	const path& working_directory();

	const path& user_local_directory();
	const path& user_home_directory();

	auto read_bytes(const path& file_path) -> std::vector<char>;
	auto read_string(const path& file_path) -> std::string;

	bool write( const path& file_path, const char* bytes, std::size_t length, bool append = false );
	bool write( const path& file_path, const std::vector<unsigned char>& bytes, bool append = false );
	bool write( const path& file_path, const std::string& str, bool append = false );
	bool create_directory( const path& dir_path, bool recursive = true );

	using list_filter = std::function<bool(const path&, bool)>;

	namespace filter {
		inline list_filter has_extension(const std::string& ext) {
			return [ext](const path& path, bool is_dir) { 
				return !is_dir && path.extension().string() == ext; 
			};
		}

		inline list_filter all() { return [](const path& path, bool is_dir) { return true; }; }
	}

	auto list( const path& dir_path, list_filter filter = filter::all() ) -> std::vector<path>;

	auto size( const path& file_path ) -> std::size_t;
	bool exists( const path& path );
	bool is_file( const path& path );
	bool is_directory( const path& path );
	long long last_modified(const path& path);
	long long now();

}