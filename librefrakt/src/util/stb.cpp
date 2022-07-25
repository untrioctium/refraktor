#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "librefrakt/util/stb.h"

bool write_file_impl(const void* data, int w, int h, int comp, const std::string& path) {
	if (path.ends_with("png"))
		return stbi_write_png(path.c_str(), w, h, comp, data, comp * w);

	if (path.ends_with("jpg") || path.ends_with("jpeg"))
		return stbi_write_jpg(path.c_str(), w, h, comp, data, 90);

	if (path.ends_with("tga"))
		return stbi_write_tga(path.c_str(), w, h, comp, data);

	return stbi_write_bmp(path.c_str(), w, h, comp, data);
}

bool rfkt::stbi::write_file(const uchar3* data, int width, int height, const std::string& path)
{
	return write_file_impl(data, width, height, 3, path);
}

bool rfkt::stbi::write_file(const uchar4* data, int width, int height, const std::string& path)
{
	return write_file_impl(data, width, height, 4, path);
}

auto rfkt::stbi::write_memory(const uchar3* data, int width, int height, format img_format) -> std::vector<unsigned char>
{
	return std::vector<unsigned char>();
}

auto rfkt::stbi::write_memory(const uchar4* data, int width, int height) -> std::vector<unsigned char>
{
	return std::vector<unsigned char>();
}
