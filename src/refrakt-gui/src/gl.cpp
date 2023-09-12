#include <imftw/imftw.h>

#include <librefrakt/util/cuda.h>
#include <librefrakt/util/filesystem.h>

#include "gl.h"

#include <cudaGL.h>

std::pair<GLuint, CUgraphicsResource> rfkt::gl::detail::allocate_texture(std::size_t w, std::size_t h, GLenum internal_format, sampling_mode s_mode)
{
	auto ret = std::make_pair<GLuint, CUgraphicsResource>(0, nullptr);

	glGenTextures(1, &ret.first);
	glBindTexture(GL_TEXTURE_2D, ret.first);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (s_mode == sampling_mode::nearest)? GL_NEAREST : GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (s_mode == sampling_mode::nearest)? GL_NEAREST : GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	CUDA_SAFE_CALL(cuGraphicsGLRegisterImage(&ret.second, ret.first, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

	return ret;
}

void rfkt::gl::detail::deallocate_texture(GLuint tex_id, CUgraphicsResource cuda_res)
{
	if (tex_id == 0) return;

	auto deleter = [cuda_res, tex_id]() {
		CUDA_SAFE_CALL(cuGraphicsUnregisterResource(cuda_res));
		glDeleteTextures(1, &tex_id);
	};

	if (ImFtw::OnRenderingThread()) deleter();
	else ImFtw::DeferNextFrame(std::move(deleter));
}

std::pair<CUDA_MEMCPY2D, CUarray> rfkt::gl::detail::create_mapping(CUgraphicsResource cuda_res, std::size_t w, std::size_t h, std::size_t pixel_size)
{
	auto ret = std::make_pair<CUDA_MEMCPY2D, CUarray>(CUDA_MEMCPY2D(), nullptr);

	if (auto res = cuGraphicsMapResources(1, &cuda_res, 0); res == CUDA_SUCCESS || res == CUDA_ERROR_ALREADY_MAPPED) {
		CUDA_SAFE_CALL(cuGraphicsSubResourceGetMappedArray(&ret.second, cuda_res, 0, 0));
	}
	else {
		CUDA_SAFE_CALL(res);
	}

	std::memset(&ret.first, 0, sizeof(ret.first));
	ret.first.srcXInBytes = 0;
	ret.first.srcY = 0;
	ret.first.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	ret.first.srcPitch = w * pixel_size;
	ret.first.srcDevice = 0;

	ret.first.dstXInBytes = 0;
	ret.first.dstY = 0;
	ret.first.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	ret.first.dstArray = ret.second;

	ret.first.WidthInBytes = w * pixel_size;
	ret.first.Height = h;

	return ret;
}

void rfkt::gl::detail::destroy_mapping(CUgraphicsResource cuda_res, std::any&& parent)
{
	if (cuda_res == nullptr) return;

	auto deleter = [cuda_res, parent = std::move(parent)]() mutable {
		if(auto res = cuGraphicsUnmapResources(1, &cuda_res, 0); res != CUDA_SUCCESS && res != CUDA_ERROR_NOT_MAPPED) CUDA_SAFE_CALL(res);
	};

	if (ImFtw::OnRenderingThread()) deleter();
	else ImFtw::DeferNextFrame(std::move(deleter));
}