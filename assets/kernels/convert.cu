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

unsigned short half2ushort10(__half v) {
    return static_cast<unsigned short>(min(__half2float(v) * 1023.0, 1023.0f));
}

template<typename T, unsigned int N>
struct array {
    T data[N];

    __device__ T& operator[](unsigned int idx) {
        return data[idx];
    }
};

using uint32 = unsigned int;

uint32 rgba10_pack(half3 val) {
    unsigned short r = half2ushort10(val.x);
    unsigned short g = half2ushort10(val.y);
    unsigned short b = half2ushort10(val.z);
    unsigned short a = 3;

    return (a << 30) | (r << 20) | (g << 10) | b;
}

//template<typename SourceType, unsigned int SourceComponents, typename DestType, unsigned int DestComponents, bool PlanarDest>
//void convert(const array<SourceType, SourceComponents>* const restrict __in)

__global__ void convert_24(const half3* const __restrict__ in, uchar3* out, unsigned int size) {
    const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bin_idx > size) return;

    const half3 in_val = in[bin_idx];

    out[bin_idx] = uchar3{half2uchar(in_val.x), half2uchar(in_val.y), half2uchar(in_val.z)};
}

template<bool Planar>
__global__ void convert_32(const half3* const __restrict__ in, decltype(image_type<Planar>()) __restrict__ out, unsigned int size) {

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

__global__ void to_float3(const half3* const __restrict__ in, float4* const __restrict__ out, unsigned int size) {
    
        const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (bin_idx > size) return;
    
        const half3 in_val = in[bin_idx];
    
        out[bin_idx] = float4(__half2float(in_val.x), __half2float(in_val.y), __half2float(in_val.z), 1.0f);

}