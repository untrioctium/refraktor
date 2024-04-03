//#include <cuda_fp16.h>
#ifdef ROCCU_CUDA
using __half = unsigned short;

float __half2float(__half v) {
    float val;
    asm("{cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(v));
    return val;
}

__device__ __half __float2half(const float a) {
	__half val;
	asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val) : "f"(a));
	return val;
}
#endif

// the long awaited sequel to half1 and half2
struct half3 {
    __half x, y, z;
};

struct half4 {
    __half x, y, z, w;
};

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

__device__ unsigned char half2uchar(__half v) {
    return static_cast<unsigned char>(min(__half2float(v) * 255.0, 255.0f));
}

template<typename OutType>
consteval auto element_count() {
    if constexpr ( requires { OutType::w; } ) {
        return 4;
    } else {
        return 3;
    }
}

template<typename OutElem>
OutElem convert_element(__half elem) {
    if constexpr (is_same<OutElem, unsigned char>) {
        return half2uchar(elem);
    } else if constexpr (is_same<OutElem, float>) {
        return __half2float(elem);
    } else {
        //static_assert(false, "Unsupported type");
    }
}

template<typename InType, typename OutType>
    requires (is_same<InType, half3> || is_same<InType, half4>)
__global__ void convert_to(const InType* const __restrict__ in, OutType* const __restrict__ out, unsigned int size) {
    
    using out_element_type = decltype(OutType::x);
    constexpr static auto out_elements = element_count<OutType>();
    constexpr static auto in_elements = element_count<InType>();

    const unsigned int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (bin_idx >= size) return;

    const InType in_val = in[bin_idx];

    if constexpr (out_elements == 3) {
        out[bin_idx] = OutType{ convert_element<out_element_type>(in_val.x), convert_element<out_element_type>(in_val.y), convert_element<out_element_type>(in_val.z) };
    } else {
        if constexpr(in_elements == 4) {
            out[bin_idx] = OutType{ convert_element<out_element_type>(in_val.x), convert_element<out_element_type>(in_val.y), convert_element<out_element_type>(in_val.z), convert_element<out_element_type>(in_val.w) };
        } else {
            out[bin_idx] = OutType{ convert_element<out_element_type>(in_val.x), convert_element<out_element_type>(in_val.y), convert_element<out_element_type>(in_val.z), convert_element<out_element_type>(__float2half(1.0f)) };
        }
    }
}