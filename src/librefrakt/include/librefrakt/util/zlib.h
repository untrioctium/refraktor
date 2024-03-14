#include <span>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <librefrakt/util/filesystem.h>

namespace rfkt::zlib {
	std::vector<char> compress(const std::vector<char>& data, unsigned int level = 9);
	std::vector<char> compress(const void* data, std::size_t len, unsigned int level = 9);

	std::string compress_b64(const std::vector<char>& data, unsigned int level = 9);
	std::string compress_b64(const std::vector<unsigned char>& data, unsigned int level = 9);
	std::string compress_b64(const void* data, std::size_t len, unsigned int level = 9);

	std::vector<char> uncompress(const std::vector<char>& data);
	std::vector<char> uncompress(const void* data, std::size_t len);
	std::vector<char> uncompress_b64(std::string_view data);

	bool extract_zip(const rfkt::fs::path& zip_path, const rfkt::fs::path& out_path);
}

/*namespace rfkt::zlib {

	using return_t = std::optional<std::vector<char>>;

	return_t compress(std::span<const char> data, unsigned int level = 9);
	return_t uncompress(std::span<const char> data);

	bool extract_zip(const rfkt::fs::path& zip_path, const rfkt::fs::path& out_path);
}*/