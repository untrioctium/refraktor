#include <imftw/imftw.h>

#include <librefrakt/util/cuda.h>
#include <librefrakt/util/filesystem.h>

#include "gl.h"

#include <cudaGL.h>

rfkt::gl::texture::texture(std::size_t w, std::size_t h) : width_(w), height_(h)
{
	glGenTextures(1, &tex_id);
	glBindTexture(GL_TEXTURE_2D, tex_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	CUDA_SAFE_CALL(cuGraphicsGLRegisterImage(&cuda_res, tex_id, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
}

rfkt::gl::texture::~texture()
{
	if (tex_id == 0) return;

	auto deleter = [cuda_res = this->cuda_res, tex_id = this->tex_id]() {
		CUDA_SAFE_CALL(cuGraphicsUnregisterResource(cuda_res));
		glDeleteTextures(1, &tex_id);
	};

	if (ImFtw::OnRenderingThread()) deleter();
	else ImFtw::DeferNextFrame(std::move(deleter));
}

rfkt::gl::texture::cuda_map::cuda_map(texture& tex) : cuda_res(tex.cuda_res)
{
	CUDA_SAFE_CALL(cuGraphicsMapResources(1, &cuda_res, 0));
	CUDA_SAFE_CALL(cuGraphicsSubResourceGetMappedArray(&arr, cuda_res, 0, 0));

	std::memset(&copy_params, 0, sizeof(copy_params));
	copy_params.srcXInBytes = 0;
	copy_params.srcY = 0;
	copy_params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copy_params.srcPitch = tex.width() * 4;
	copy_params.srcDevice = 0;

	copy_params.dstXInBytes = 0;
	copy_params.dstY = 0;
	copy_params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy_params.dstArray = arr;

	copy_params.WidthInBytes = tex.width() * 4;
	copy_params.Height = tex.height();

}

rfkt::gl::texture::cuda_map::~cuda_map()
{
	if (cuda_res == nullptr) return;

	auto deleter = [cuda_res = &this->cuda_res]() mutable {
		CUDA_SAFE_CALL(cuGraphicsUnmapResources(1, cuda_res, 0));
	};

	if (ImFtw::OnRenderingThread()) deleter();
	else ImFtw::DeferNextFrame(std::move(deleter));
}