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