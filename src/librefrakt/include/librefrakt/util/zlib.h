#include <vector>

namespace rfkt::zlib {
	std::vector<char> compress(const std::vector<char>& data, unsigned int level = 9);
	std::vector<char> compress(const void* data, std::size_t len, unsigned int level = 9);

	std::vector<char> uncompress(const std::vector<char>& data);
	std::vector<char> uncompress(const void* data, std::size_t len);
}