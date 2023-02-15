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


template<typename FloatT>
struct vec2 {
	FloatT x, y;
};

template<typename FloatT>
struct vec3 {
	FloatT x, y, z;
};

consteval Real operator "" _r(long double v) {
	return static_cast<Real>(v);
}
