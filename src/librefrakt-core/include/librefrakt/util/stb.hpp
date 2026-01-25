#include <vector>
#include <string>

#include <librefrakt/vector_types.hpp>

namespace rfkt::stbi {

	enum class format {
		png,
		bmp,
		tga,
		jpg,
	};

	bool write_file(const uchar3* data, int width, int height, const std::string& path);
	bool write_file(const uchar4* data, int width, int height, const std::string& path);

	auto write_memory(const uchar3* data, int width, int height, format img_format)->std::vector<std::byte>;
	auto write_memory(const uchar4* data, int width, int height)->std::vector<std::byte>;
}