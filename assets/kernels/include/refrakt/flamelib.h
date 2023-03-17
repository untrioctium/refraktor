#ifdef DOUBLE_PRECISION 
	using Real = double;
	#define M_PI 3.14159265358979323846264338327950288419
	#define M_1_PI (1.0/3.14159265358979323846264338327950288419)
	#define M_EPS (1e-10)
	#define INTERP(a, b, mix) ((a) * (1.0 - (mix)) + (b) * (mix))
	#define __sincos sincos
	#define __pow pow
	#define __fma fmaf
	#define badvalue(x) (((x)!=(x))||((x)>1e10)||((x)<-1e10))
#else
	using Real = float;
	#define M_PI 3.14159265358979323846264338327950288419f
	#define M_1_PI (1.0f/3.14159265358979323846264338327950288419f)
	#define M_PI_2 (3.14159265358979323846264338327950288419f/2.0f)
	#define M_2_PI (2.0f/3.14159265358979323846264338327950288419f)
	#define M_EPS (1e-10f)
	#define INTERP(a, b, mix) ((a) * (1.0f - (mix)) + (b) * (mix))
	#define __sincos __sincosf
	#define __pow __powf
	#define __fma fma
	#define badvalue(x) (((x)!=(x))||((x)>1e10f)||((x)<-1e10f))
#endif

consteval Real operator "" _r(long double v) {
	return static_cast<Real>(v);
}

using uint64 = unsigned long long int;
using uint32 = unsigned int;
using uint16 = unsigned short;
using uint8 = unsigned char;
using int64 = long long int;
using int32 = int;
using int16 = short;
using int8 = char;

constexpr static uint32 palette_channel_size = 256;
constexpr static uint32 num_channels = 3;
constexpr static uint32 palette_size = num_channels * palette_channel_size;

template<typename FloatT>
struct vec2 {
	FloatT x, y;

	vec2& operator+=(const vec2& other) {
		x += other.x;
        y += other.y;
        return *this;
	}

	vec2& operator-=(const vec2& other) {
		x += other.x;
        y += other.y;
        return *this;
	}
};
static_assert(sizeof(vec2<float>) == sizeof(float) * 2);

template<typename FloatT>
struct vec3 : public vec2<FloatT> {
	FloatT z;

	vec2<FloatT>& as_vec2() { return *reinterpret_cast<vec2<FloatT>*>(this); }
};
static_assert(sizeof(vec3<float>) == sizeof(float) * 3);

template<typename FloatT>
struct vec4 : public vec3<FloatT> {
	FloatT w;

	vec3<FloatT>& as_vec3() { return *reinterpret_cast<vec3<FloatT>*>(this); }
};
static_assert(sizeof(vec4<float>) == sizeof(float) * 4);

namespace flamelib {
	__device__ constexpr auto warp_size() { return 32; }

	__device__ inline void sync_warp() { __syncwarp(); }
	__device__ inline void sync_block() { __syncthreads(); }
	
		template<typename T>
	__device__ inline T warp_reduce( T thread_val ) {
		
		#if __CUDA_ARCH__ >= 800
			return __reduce_add_sync(0xffffffff, thread_val);
		#else
			for (int i=16; i>=1; i/=2)
				thread_val += __shfl_down_sync(0xffffffff, thread_val, i);
			
			return thread_val;
		#endif
	}
	
	__device__ inline bool is_warp_leader() { return threadIdx.x % warp_size() == 0; }
	__device__ inline bool is_block_leader() { return threadIdx.x == 0; }
	__device__ inline bool is_grid_leader() { return threadIdx.x == 0 && blockIdx.x == 0; };
	
	__device__ inline auto warp_rank() { return threadIdx.x % warp_size(); }
	__device__ inline auto block_rank() { return threadIdx.x; }
	__device__ inline auto grid_rank() { return threadIdx.x + blockIdx.x * blockDim.x; }
	
	__device__ inline auto warp_start_in_block() { return block_rank() - warp_rank(); }

	__device__ inline auto time() {
		uint64 tmp;
		asm volatile("mov.u64 %0, %globaltimer;":"=l"(tmp)::);
		return tmp;
	}

	template<typename FloatT, uint64 ThreadsPerBlock>
	struct iterators_t {
		FloatT x[ThreadsPerBlock];
		FloatT y[ThreadsPerBlock];
		FloatT color[ThreadsPerBlock];
	};

	template<typename FloatT, typename RandCtx, uint64 ThreadsPerBlock>
	struct thread_states_t {
		iterators_t<FloatT, ThreadsPerBlock> iterators;
		RandCtx rand_states[ThreadsPerBlock];
		uint16 shuffle_vote[ThreadsPerBlock];
		uint16 shuffle[ThreadsPerBlock];
		uint8 xform_vote[ThreadsPerBlock];
	};

	template<typename FlameT, typename FloatT, typename RandCtx, uint64 ThreadsPerBlock>
	struct __align__(16) shared_state_tmpl {
		thread_states_t<FloatT, RandCtx, ThreadsPerBlock> ts;
		uchar3 palette[palette_channel_size];
		
		vec2<FloatT> antialiasing_offsets;
		unsigned long long tss_quality;
		unsigned long long tss_passes;
		uint64 tss_start;
		bool should_bail;

		FlameT flame;
	};
}

// operator overloads

template<typename T>
__device__ vec2<T> operator+(const vec2<T>& a, const vec2<T>& b) noexcept {
    return {a.x + b.x, a.y + b.y};
}

template<typename T>
__device__ vec2<T> operator-(const vec2<T>& a, const vec2<T>& b) noexcept {
    return {a.x - b.x, a.y - b.y};
}

template<typename T>
__device__ vec3<T> operator+(const vec3<T>& a, const vec3<T>& b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z - b.z};
}

template<typename T>
__device__ vec3<T> operator-(const vec3<T>& a, const vec3<T>& b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template<typename T>
__device__ vec2<T> operator*(const vec2<T>& a, T scalar) noexcept {
    return {a.x * scalar, a.y * scalar};
}

template<typename T>
__device__ vec2<T> operator*(T scalar, const vec2<T>& a) noexcept {
    return {scalar * a.x, scalar * a.y};
}

template<typename T>
__device__ vec2<T> operator/(const vec2<T>& a, T scalar) noexcept {
    return {a.x / scalar, a.y / scalar};
}

template<typename T>
 __device__ vec3<T> operator/(const vec3<T>& a, T scalar) noexcept {
    return {a.x / scalar, a.y / scalar, a.z / scalar};
 }

// functions

#define MAKE_SIMPLE_OVERLOAD(name, fname, dname) \
	template<typename T> \
	__device__ T name(T a) noexcept { \
	    if constexpr(is_same<T, float>) \
			return ::fname(a); \
		else return ::dname(a); \
	}\
    template<typename T> \
    __device__ vec2<T> name(const vec2<T>& a) noexcept { \
        return {flamelib::name(a.x), flamelib::name(a.y)}; \
    } \
    template<typename T> \
    __device__ vec3<T> name(const vec3<T>& a) noexcept { \
        return {flamelib::name(a.x), flamelib::name(a.y), flamelib::name(a.z)}; \
    }

namespace flamelib {

	namespace math {
		Real pi = 3.1415926535897932384626433832795028841971693_r;
		Real inv_pi = 0.318309886183790671537767526745028724_r;
	}

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

	template<typename T, typename U>
	static constexpr bool is_same = is_same_t<T, U>::value;

	// trigonometric and hyperbolic functions
	MAKE_SIMPLE_OVERLOAD(sin, sinf, sin);
	MAKE_SIMPLE_OVERLOAD(sinpi, sinpif, sinpi);
	MAKE_SIMPLE_OVERLOAD(cos, cosf, cos);
	MAKE_SIMPLE_OVERLOAD(cospi, cospif, cospi);
	MAKE_SIMPLE_OVERLOAD(tan, tanf, tan);
	MAKE_SIMPLE_OVERLOAD(tanpi, tanpif, tanpi);
	MAKE_SIMPLE_OVERLOAD(sinh, sinhf, sinh);
	MAKE_SIMPLE_OVERLOAD(cosh, coshf, cosh);
	MAKE_SIMPLE_OVERLOAD(tanh, tanhf, tanh);
	MAKE_SIMPLE_OVERLOAD(asin, asinf, asin);
	MAKE_SIMPLE_OVERLOAD(acos, acosf, acos);
	MAKE_SIMPLE_OVERLOAD(atan, atanf, atan);
	MAKE_SIMPLE_OVERLOAD(asinh, asinhf, asinh);
	MAKE_SIMPLE_OVERLOAD(acosh, acoshf, acosh);
	MAKE_SIMPLE_OVERLOAD(atanh, atanhf, atanh);

	template<typename T>
	T atan2(T a, T b) noexcept {
		if constexpr (is_same<T, float>)
			return ::atan2f(a, b);
		else return ::atan2(a, b);
    }

	template<typename T>
	T hypot(T a, T b) noexcept {
		if constexpr (is_same<T, float>)
			return ::hypotf(a, b);
		else return ::hypot(a, b);
    }

	template<typename T>
	T pow(T a, T b) noexcept {
		if constexpr (is_same<T, float>)
			return ::powf(a, b);
		else return ::pow(a, b);
    }

	template<typename T>
	T modf(T a, T b) noexcept {
		if constexpr (is_same<T, float>)
			return ::fmodf(a, b);
		else return ::modf(a, b);
    }

	// exponential and logarithmic functions
	MAKE_SIMPLE_OVERLOAD(abs, fabs, abs);
	MAKE_SIMPLE_OVERLOAD(sqrt, sqrtf, sqrt);
	MAKE_SIMPLE_OVERLOAD(exp, expf, exp);
	MAKE_SIMPLE_OVERLOAD(log, logf, log);
	MAKE_SIMPLE_OVERLOAD(exp2, exp2f, exp2);
	MAKE_SIMPLE_OVERLOAD(log2, log2f, log2);
	MAKE_SIMPLE_OVERLOAD(exp10, exp10f, exp10);
	MAKE_SIMPLE_OVERLOAD(log10, log10f, log10);

	MAKE_SIMPLE_OVERLOAD(trunc, truncf, trunc);
	MAKE_SIMPLE_OVERLOAD(floor, floorf, floor);

	template<typename T>
	__device__ vec2<T> sincos(T value) noexcept {
		vec2<T> result = {};
		if constexpr(is_same<T, float>)
			sincosf(value, &result.x, &result.y);
		else 
			sincos(value, &result.x, &result.y);
		return result;
	}

	template<typename T>
	__device__ vec2<T> cossin(T value) noexcept {
		vec2<T> result = {};
		if constexpr(is_same<T, float>)
			sincosf(value, &result.y, &result.x);
		else 
			sincos(value, &result.y, &result.x);
		return result;
	}

	template<typename T>
	__device__ vec2<T> sincospi(T value) noexcept {
		vec2<T> result = {};
		if constexpr(is_same<T, float>)
			sincospif(value, &result.x, &result.y);
		else 
			sincospi(value, &result.x, &result.y);
		return result;
	}

	template<typename T>
	__device__ vec2<T> cossinpi(T value) noexcept {
		vec2<T> result = {};
		if constexpr(is_same<T, float>)
			sincospif(value, &result.y, &result.x);
		else 
			sincospi(value, &result.y, &result.x);
		return result;
	}

	template<typename T>
	__device__ vec2<T> elmul(const vec2<T>& a, const vec2<T>& b) noexcept {
		return {a.x * b.x, a.y * b.y};
	}

	template<typename T>
	__device__ T fma(T a, T b, T c) noexcept {
		if constexpr (is_same<T, float>)
			return ::fmaf(a, b, c);
		else return ::fma(a, b, c);
	}

	template<typename T>
	__device__ vec2<T> fma(const vec2<T>& a, T b, const vec2<T>& c) noexcept {
		return {flamelib::fma(a.x, b, c.x), flamelib::fma(a.y, b, c.y)};
	}

	template<typename T>
	__device__ vec2<T> fma(T a, const vec2<T>& b, const vec2<T>& c) noexcept {
		return {flamelib::fma(a, b.x, c.x), flamelib::fma(a, b.y, c.y)};
	}

	template<typename T>
	__device__ T eps(T a) noexcept {
		return (a == 0.0)? M_EPS: a;
	}

	template<typename T>
	__device__ vec2<T> eps(const vec2<T>& a) noexcept {
        return {flamelib::eps(a.x), flamelib::eps(a.y)};
    }
}