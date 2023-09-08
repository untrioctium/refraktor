//#include <cuda_fp16.h>
using __half = unsigned short;


// the long awaited sequel to half1 and half2
struct half3 {
    __half x, y, z;
};

template<bool Planar>
consteval auto image_type() {
    if constexpr (Planar) {
        return (unsigned char*)(nullptr);
    } else {
        return (uchar4*)(nullptr);
    }
}

float __half2float(__half v) {
    float val;
    asm("{cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(v));
    return val;
}

unsigned char half2uchar(__half v) {
    return static_cast<unsigned char>(min(__half2float(v) * 255.0, 255.0f));
}

template<bool Planar>
__global__ void convert(const half3* const __restrict__ in, decltype(image_type<Planar>()) __restrict__ out, unsigned int size) {

    const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (bin_idx > size) return;

    const half3 in_val = in[bin_idx];

    if constexpr (Planar) {
        out[bin_idx] = half2uchar(in_val.x);
        out[bin_idx + size] = half2uchar(in_val.y);
        out[bin_idx + 2 * size] = half2uchar(in_val.z);
        out[bin_idx + 3 * size] = 255;
    } else {
        out[bin_idx] = uchar4{half2uchar(in_val.x), half2uchar(in_val.y), half2uchar(in_val.z), 255};
    }
}

unsigned int half2_10bit(__half v) {
    return static_cast<unsigned int>(min(__half2float(v) * 1023.0f, 1023.0f));
}

__global__ void to_10bit(const half3* const __restrict__ in, unsigned int* const __restrict__ out, unsigned int size) {

    const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bin_idx > size) return;

    const half3 in_val = in[bin_idx];

    out[bin_idx] = half2_10bit(in_val.x) | (half2_10bit(in_val.y) << 10) | (half2_10bit(in_val.z) << 20) | 0b11 << 30;;
}

__global__ void split(const float4* const __restrict__ bins, float3* const __restrict__ rgb, float3* const __restrict alpha, unsigned int size) {
    
        const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (bin_idx > size) return;
    
        const float4 bin_val = bins[bin_idx];

        if(bin_val.w == 0.0) {
		    rgb[bin_idx] = {0, 0, 0};
            alpha[bin_idx] = {0, 0, 0};
		    return;
	    }
    
        rgb[bin_idx] = float3(bin_val.x, bin_val.y, bin_val.z);
        alpha[bin_idx] = float3(bin_val.w, bin_val.w, bin_val.w);
}

__global__ void join(const float3* const __restrict__ rgb, const float3* const __restrict__ alpha, float4* const __restrict out, unsigned int size) {
    
        const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (bin_idx > size) return;
    
        const float3 rgb_val = rgb[bin_idx];
        const float3 alpha_val = alpha[bin_idx];

        out[bin_idx] = float4(rgb_val.x, rgb_val.y, rgb_val.z, alpha_val.x);
}