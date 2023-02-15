#include <cuda_fp16.h>

#define DEBUG(...) if(threadIdx.x == 0 && blockIdx.x == 0) {printf(__VA_ARGS__);}

struct half3 {
	__half x, y, z;
};

__global__ void tonemap(const float4* __restrict__ bins, const ushort4* __restrict__ accumulator, half3* __restrict__ image, unsigned int dims_x, unsigned int dims_y, float gamma, float scale_constant, float brightness, float vibrancy) {

	//DEBUG("%dx%d (%f,%f,%f,%f)\n", dims_x, dims_y, gamma, scale_constant, brightness, vibrancy);

	const half3 background = { 0, 0, 0 };
	const uint2 pos = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y
	};

	if (pos.x >= dims_x || pos.y >= dims_y) return;

	const unsigned int bin_idx = (pos.y) * dims_x + pos.x;
	float4 col = bins[bin_idx];

	if(accumulator != nullptr) {
		ushort4 accum = accumulator[bin_idx];
		col.x += accum.x / 255.0f;
		col.y += accum.y / 255.0f;
		col.z += accum.z / 255.0f;
		col.w += accum.w / 255.0f;
	}

	if(col.w == 0.0) {
		image[bin_idx] = background;
		return;
	}

	col.w += 1;
	const float factor = (col.w == 0.0f)? 0.0f : 0.5f * brightness * logf(1.0f + col.w * scale_constant) * 0.434294481903251827651128918916f / (col.w);
	col.x *= factor; col.y *= factor; col.z *= factor; col.w *= factor;

	const float inv_gamma = 1.0f / gamma;
	const float z = pow(col.w, inv_gamma);
	const float gamma_factor = z / col.w;

	col.x *= gamma_factor; 
	col.y *= gamma_factor;
	col.z *= gamma_factor;
	//col.w *= gamma_factor;

	#define interp(left, right, mix) ((left) * (1.0f - (mix)) + (right) * (mix))

	image[bin_idx] = { __float2half(col.x), __float2half(col.y), __float2half(col.z) };
}