#include <roccu_vector_types.h>
#include <vector>
#include <string>

namespace rfkt::stbi {

	enum class format {
		png,
		bmp,
		tga,
		jpg,
	};

	bool write_file(const uchar3* data, int width, int height, const std::string& path);
	bool write_file(const uchar4* data, int width, int height, const std::string& path);

	auto write_memory(const uchar3* data, int width, int height, format img_format)->std::vector<unsigned char>;
	auto write_memory(const uchar4* data, int width, int height)->std::vector<unsigned char>;
}