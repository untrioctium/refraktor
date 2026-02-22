#define DEBUG(fmt, ...) if(threadIdx.x == 0) {printf("block %d: " fmt, blockIdx.x, __VA_ARGS__);}

#ifdef ROCCU_CUDA
using __half = unsigned short;

__device__ __half __float2half(const float a) {
	__half val;
	asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val) : "f"(a));
	return val;
}

float __half2float(__half v) {
    float val;
    asm("{cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(v));
    return val;
}
#endif

struct half3 {
	__half x, y, z;
};

struct half4 {
	__half x, y, z, w;
};

__device__ float lerp(float a, float b, float t) {
	return a + t * (b - a);
}

namespace detail {
    template<bool v>
    struct boolean_constant {
        static constexpr bool value = v;
    };

    using true_type = boolean_constant<true>;
    using false_type = boolean_constant<false>;

    template<class T, class U>
    struct is_same_t : false_type {};
    
    template<class T>
    struct is_same_t<T, T> : true_type {};
}

template<typename T, typename U>
static constexpr bool is_same = detail::is_same_t<T, U>::value;

template<typename OutPixelType>
__global__ void tonemap(const float4* __restrict__ cold_bins, const half4* __restrict__ hot_bins, OutPixelType* __restrict__ image, unsigned int img_width, unsigned int img_size, unsigned int scaling, float gamma, float scale_constant, float brightness, float vibrancy, bool hdr) {

	constexpr static bool DoAlpha = requires { OutPixelType::w; };

	constexpr static auto to_out_channel_type = [](float value) {
		if constexpr (is_same<OutPixelType, half3> || is_same<OutPixelType, half4>) {
			return __float2half(value);
		} else {
			return value;
		}
	};

	auto img_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (img_idx >= img_size) return;

	auto img_x = img_idx % img_width;
	auto img_y = img_idx / img_width;

	const auto bins_width = img_width * scaling;
	const auto bins_x = img_x * scaling;
	const auto bins_y = img_y * scaling;

	float4 col = {0.0f, 0.0f, 0.0f, 0.0f};

	for(unsigned int dy = 0; dy < scaling; dy++) {
		for(unsigned int dx = 0; dx < scaling; dx++) {
			auto bin_idx_local = bins_x + dx + (bins_y + dy) * bins_width;
			col.x += __half2float(hot_bins[bin_idx_local].x) + cold_bins[bin_idx_local].x;
			col.y += __half2float(hot_bins[bin_idx_local].y) + cold_bins[bin_idx_local].y;
			col.z += __half2float(hot_bins[bin_idx_local].z) + cold_bins[bin_idx_local].z;
			col.w += __half2float(hot_bins[bin_idx_local].w) + cold_bins[bin_idx_local].w;
		}
	}

	if(col.w == 0.0) {
		auto zero_value = to_out_channel_type(0.0);
		if constexpr (DoAlpha) {
			image[img_idx] = { zero_value, zero_value, zero_value, zero_value };
		} else {
			image[img_idx] = { zero_value, zero_value, zero_value };
		}
		return;
	}

	//col.w += 1;
	const float input_density = col.w;
	const float factor = (col.w == 0.0f)? 0.0f : 0.5f * brightness * logf(1.0f + col.w * scale_constant) * 0.434294481903251827651128918916f / (col.w);
	col.x *= factor; col.y *= factor; col.z *= factor; col.w *= factor;

	const double inv_gamma = 1.0 / gamma;
	const double z = pow((double) col.w, inv_gamma);
	const double gamma_factor = z / col.w;

	col.x = lerp(powf(col.x, inv_gamma), col.x * gamma_factor, vibrancy);
	col.y = lerp(powf(col.y, inv_gamma), col.y * gamma_factor, vibrancy);
	col.z = lerp(powf(col.z, inv_gamma), col.z * gamma_factor, vibrancy);

	if (not hdr) {
		col.x = __saturatef(col.x);
		col.y = __saturatef(col.y);
		col.z = __saturatef(col.z);
	} else {
		constexpr static float hdr_boost = 1.5f;
		col.x *= hdr_boost;
		col.y *= hdr_boost;
		col.z *= hdr_boost;
	}

	if constexpr (DoAlpha) {
		image[img_idx] = { to_out_channel_type(col.x), to_out_channel_type(col.y), to_out_channel_type(col.z), to_out_channel_type(1.0f) };
	} else {
		image[img_idx] = { to_out_channel_type(col.x), to_out_channel_type(col.y), to_out_channel_type(col.z) };
	}
}