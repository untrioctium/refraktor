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

template<typename OutPixelType, unsigned int Granularity>
__global__ void tonemap(const float4* __restrict__ bins, OutPixelType* __restrict__ image, unsigned int size, double average_density, double max_density, float gamma, float scale_constant, float brightness, float vibrancy) {

	constexpr static auto MaxPower = Granularity * 24;
	const auto log_div = Granularity / log(2.0f);

	//DEBUG("%dx%d (%f,%f,%f,%f)\n", dims_x, dims_y, gamma, scale_constant, brightness, vibrancy);
	constexpr static bool DoAlpha = requires { OutPixelType::w; };

	constexpr static auto to_out_channel_type = [](float value) {
		if constexpr (is_same<OutPixelType, half3> || is_same<OutPixelType, half4>) {
			return __float2half(value);
		} else {
			return value;
		}
	};

	auto bin_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (bin_idx >= size) return;

	float4 col = bins[bin_idx];

	if(col.w == 0.0) {
		auto zero_value = to_out_channel_type(0.0);
		if constexpr (DoAlpha) {
			image[bin_idx] = { zero_value, zero_value, zero_value, zero_value };
		} else {
			image[bin_idx] = { zero_value, zero_value, zero_value };
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

	col.x = __saturatef(lerp(powf(col.x, inv_gamma), col.x * gamma_factor, vibrancy));
	col.y = __saturatef(lerp(powf(col.y, inv_gamma), col.y * gamma_factor, vibrancy));
	col.z = __saturatef(lerp(powf(col.z, inv_gamma), col.z * gamma_factor, vibrancy));
	if constexpr (DoAlpha) {
		//col.w = input_density / max_density;
		col.w = __saturatef(log(input_density) * log_div / MaxPower);
	}

	if constexpr (DoAlpha) {
		image[bin_idx] = { to_out_channel_type(col.x), to_out_channel_type(col.y), to_out_channel_type(col.z), to_out_channel_type(col.w) };
	} else {
		image[bin_idx] = { to_out_channel_type(col.x), to_out_channel_type(col.y), to_out_channel_type(col.z) };
	}
}

float4 rgb_to_hsv(float4 rgb) {
	float r = rgb.x;
	float g = rgb.y;
	float b = rgb.z;

	float h, s, v;

	float max = fmaxf(r, fmaxf(g, b));
	float min = fminf(r, fminf(g, b));
	float delta = max - min;

	v = max;
	s = max == 0 ? 0 : delta / max;

	if(s == 0.0) {
		h = 0.0;
	} else {
		h = 60.0 * (max == r ? (g - b) / delta : max == g ? 2 + (b - r) / delta : 4 + (r - g) / delta);
		if(h < 0) h += 360;
	}

	return { h, s, v, rgb.w };
}

float4 hsv_to_rgb(float4 hsv) {
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;

	while(h < 0) h += 360;
	
	if(s == 0.0) {
		return { v, v, v, hsv.w };
	} else {
		h = fmodf(h, 360.0f) / 60.0f;
		int i = (int) h;
		float f = h - i;
		float p = v * (1 - s);
		float q = v * (1 - s * f);
		float t = v * (1 - s * (1 - f));

		switch(i) {
			case 0: return { v, t, p, hsv.w };
			case 1: return { q, v, p, hsv.w };
			case 2: return { p, v, t, hsv.w };
			case 3: return { p, q, v, hsv.w };
			case 4: return { t, p, v, hsv.w };
			default: return { v, p, q, hsv.w };
		}
	}
}

template<typename InPixelType, unsigned int Granularity>
__global__ void density_hdr(const InPixelType* __restrict__ in, float4* __restrict__ out, unsigned int size, unsigned long long* histogram) {

	constexpr static auto MaxPower = Granularity * 24;

	__shared__ float histo_cdf[MaxPower + 1];

	if(threadIdx.x == 0) {
		float histo_sum = 0;
		for(int i = 0; i <= MaxPower; i++) {
			histo_sum += static_cast<float>(histogram[i]);
		}
		for(int i = 0; i <= MaxPower; i++) {
			histo_cdf[i] = static_cast<float>(histogram[i]) / histo_sum;

			if(i > 0) {
				histo_cdf[i] += histo_cdf[i - 1];
			}
		}
	}
	__syncthreads();

	auto bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (bin_idx >= size) return;

	constexpr static auto from_in_channel_type = [](auto value) {
		if constexpr (is_same<InPixelType, half3> || is_same<InPixelType, half4>) {
			return __half2float(value);
		} else {
			return value;
		}
	};

	float4 col = { from_in_channel_type(in[bin_idx].x), from_in_channel_type(in[bin_idx].y), from_in_channel_type(in[bin_idx].z), from_in_channel_type(in[bin_idx].w) };

	//out[bin_idx] = {col.w, col.w, col.w, 1.0f};
	//return;

	//if(col.w < 0.00001f) {
	//	out[bin_idx] = { 0, 0, 0, 1.0f };
	//	return;
	//}

	auto power = col.w * MaxPower + 1;
	auto low_bin = static_cast<int>(power);
	auto high_bin = low_bin + 1;
	auto fract_bin = power - low_bin;

	auto cdf_val = lerp(histo_cdf[low_bin], histo_cdf[high_bin], fract_bin);
	//if(cdf_val <= 0.5) {
	//	out[bin_idx] = { cdf_val * 2, cdf_val * 2, cdf_val * 2, 1.0f };
	//} else {
	//	cdf_val = 1 - (cdf_val - 0.5) * 2;
	//	out[bin_idx] = { cdf_val, cdf_val, cdf_val, 1.0f};
	//
	//}
	//out[bin_idx] = { cdf_val, cdf_val, cdf_val, 1.0f };
	//return;

	//col.x = powf(col.x, 2.2f);
	//col.y = powf(col.y, 2.2f);
	//col.z = powf(col.z, 2.2f);

	//if(cdf_val >= 0.5) {
		float4 hsv = rgb_to_hsv(col);
		float add_value = cdf_val / .5;
		//DEBUG("cdf: %f add_value: %f\n", cdf_val, add_value);
		hsv.z *= add_value * add_value;
		col = hsv_to_rgb(hsv);
	//}

	//col = hsv_to_rgb(rgb_to_hsv(col));

	out[bin_idx] = { col.x, col.y, col.z, 1.0f };

}