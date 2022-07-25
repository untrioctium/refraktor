#ifdef DOUBLE_PRECISION 
	using Real = double;
	using vec2 = double2;
	using vec3 = double3;
	using vec4 = double4;
	#define M_PI 3.14159265358979323846264338327950288419
	#define M_1_PI (1.0/3.14159265358979323846264338327950288419)
	#define M_EPS (1e-10)
	#define INTERP(a, b, mix) ((a) * (1.0 - (mix)) + (b) * (mix))
	#define __sincos sincos
	#define __pow pow
#else
	using Real = float;
	using vec2 = float2;
	using vec3 = float3;
	using vec4 = float4;
	#define M_PI 3.14159265358979323846264338327950288419f
	#define M_1_PI (1.0f/3.14159265358979323846264338327950288419f)
	#define M_PI_2 (2.0f/3.14159265358979323846264338327950288419f)
	#define M_2_PI (2.0f/3.14159265358979323846264338327950288419f)
	#define M_EPS (1e-10)
	#define INTERP(a, b, mix) ((a) * (1.0f - (mix)) + (b) * (mix))
	#define __sincos __sincosf
	#define __pow __powf
#endif

#define badvalue(x) (((x)!=(x))||((x)>1e10)||((x)<-1e10))
