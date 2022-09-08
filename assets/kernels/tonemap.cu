#define DEBUG(...) if(threadIdx.x == 0 && blockIdx.x == 0) {printf(__VA_ARGS__);}

template<bool Planar>
constexpr auto image_type() {
	if constexpr(Planar) 
		return static_cast<unsigned char>(0);
	else 
		return uchar4{0};
}

template<bool Planar>
__global__ void tonemap(const float4* __restrict__ bins, decltype(image_type<Planar>())*  __restrict__ image, unsigned int dims_x, unsigned int dims_y, float gamma, float scale_constant, float brightness, float vibrancy) {

	//DEBUG("%dx%d (%f,%f,%f,%f)\n", dims_x, dims_y, gamma, scale_constant, brightness, vibrancy);

	const uchar4 background = { 0, 0, 0, 255 };
	const uint2 pos = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y
	};

	if (pos.x >= dims_x || pos.y >= dims_y) return;

	const unsigned int bin_idx = (pos.y) * dims_x + pos.x;
	float4 col = bins[bin_idx];

	if(col.w == 0.0) {
		if constexpr(Planar) {
			const auto channel_stride = dims_x * dims_y;
			image[bin_idx] = background.x;
			image[channel_stride + bin_idx] = background.y;
			image[2 * channel_stride + bin_idx] = background.z;
			image[3 * channel_stride + bin_idx] = background.w;

		} else {
			image[bin_idx] = background;
		}
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
	col.w *= gamma_factor;

	#define interp(left, right, mix) ((left) * (1.0f - (mix)) + (right) * (mix))

	if constexpr(Planar) {
		const auto channel_stride = dims_x * dims_y;
		image[bin_idx] = min(255.0f, col.x * 255.0f);
		image[channel_stride + bin_idx] = min(255.0f, col.y * 255.0f);
		image[2 * channel_stride + bin_idx] = min(255.0f, col.z * 255.0f);
		image[3 * channel_stride + bin_idx] = 255;
	}
	else {
		image[bin_idx] = {
			(unsigned char) min(255.0f, col.x * 255.0f),
			(unsigned char) min(255.0f, col.y * 255.0f),
			(unsigned char) min(255.0f, col.z * 255.0f),
			255//(unsigned char) min(255.0f, col.w * 255.0f)
		};
	}
}