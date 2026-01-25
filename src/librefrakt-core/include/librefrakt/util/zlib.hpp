#include <vector>
#include <span>
#include <string>
#include <string_view>

namespace rfkt::zlib {
	std::vector<char> compress(std::span<const char> data, unsigned int level = 9);
	std::vector<char> compress(std::span<const unsigned char> data, unsigned int level = 9);
	std::vector<char> compress(const void* data, std::size_t len, unsigned int level = 9);

	std::string compress_b64(std::span<const char> data, unsigned int level = 9);
	std::string compress_b64(std::span<const unsigned char> data, unsigned int level = 9);
	std::string compress_b64(const void* data, std::size_t len, unsigned int level = 9);

	std::vector<char> uncompress(std::span<const char> data);
	std::vector<char> uncompress(std::span<const unsigned char> data);
	std::vector<char> uncompress(const void* data, std::size_t len);
	std::vector<char> uncompress_b64(std::string_view data);
}