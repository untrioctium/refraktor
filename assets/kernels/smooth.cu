__device__ __inline__ void accum_atomic(float4* mem, const float4 val) {
	float* f_ptr = (float*)mem;
	atomicAdd(f_ptr    , val.x);
	atomicAdd(f_ptr + 1, val.y);
	atomicAdd(f_ptr + 2, val.z);
	atomicAdd(f_ptr + 3, val.w);
}

__device__ __inline__ float4 mult_scalar(const float4& v, float scalar) {
	return make_float4(
		v.x * scalar,
		v.y * scalar,
		v.z * scalar,
		v.w * scalar
	);
}

__global__ void smooth(
	const float4* image_in,
	float4* image_out,
	unsigned int image_w,
	unsigned int image_h,

	int kernel_radius,
	int kernel_min,
	float kernel_curve
) {

	auto x_idx = threadIdx.x + blockDim.x * blockIdx.x;
	auto y_idx = threadIdx.y + blockDim.y * blockIdx.y;

	if (x_idx >= image_w || y_idx >= image_h) return;

	float4 in_point = image_in[y_idx * image_w + x_idx];

	int radius = max(
		kernel_min,
		min(
			kernel_radius,
			int(
				floor(kernel_radius / pow(in_point.w, kernel_curve))
			)
		)
	);

	if (in_point.w == 0.0f || radius == 0) {
		accum_atomic(image_out + y_idx * image_w + x_idx, in_point);
		return;
	}

	float norm_factor = 0.63661977236f / float(radius * radius);

	for (int dy = -radius; dy <= radius; dy++) {
		for (int dx = -radius; dx <= radius; dx++) {
			int cur_x = x_idx + dx;
			int cur_y = y_idx + dy;

			if (cur_x < 0 || cur_y < 0 || cur_x >= image_w || cur_y >= image_h) continue;

			float norm_distance = float(dy * dy + dx * dx) / (radius * radius);

			if (norm_distance >= 1.0f) continue;

			accum_atomic(image_out + cur_y * image_w + cur_x, mult_scalar(in_point, (1.0f - norm_distance) * norm_factor));
		}
	}

}