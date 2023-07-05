#define DEBUG(...) if(threadIdx.x == 0 && blockIdx.x == 0) {printf(__VA_ARGS__);}

using __half = unsigned short;
struct half3 {
	__half x, y, z;
};

__half __float2half(const float a) {
	__half val;
	asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val) : "f"(a));
	return val;
}

__global__ void tonemap(const float4* __restrict__ bins, half3* __restrict__ image, unsigned int size, float gamma, float scale_constant, float brightness, float vibrancy) {

	//DEBUG("%dx%d (%f,%f,%f,%f)\n", dims_x, dims_y, gamma, scale_constant, brightness, vibrancy);

	const half3 background = { 0, 0, 0 };
	auto bin_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (bin_idx > size) return;

	float4 col = bins[bin_idx];

	if(col.w == 0.0) {
		image[bin_idx] = background;
		return;
	}

	if(col.x > 65536 * 256 || col.y > 65536 * 256 || col.z > 65536 * 256 || col.w > 65536) {
		printf("clip %f %f %f %f\n", col.x, col.y, col.z, col.w);
	}

	//col.w += 1;
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