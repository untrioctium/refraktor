#pragma once

template<typename T>
struct vec2_base {
	T x, y;
};

template<typename T>
struct vec3_base {
	T x, y, z;
};

template<typename T>
struct vec4_base {
	T x, y, z, w;
};

using int2 = vec2_base<int>;
using int3 = vec3_base<int>;
using int4 = vec4_base<int>;

using float2 = vec2_base<float>;
using float3 = vec3_base<float>;
using float4 = vec4_base<float>;

using double2 = vec2_base<double>;
using double3 = vec3_base<double>;
using double4 = vec4_base<double>;

using uint2 = vec2_base<unsigned int>;
using uint3 = vec3_base<unsigned int>;
using uint4 = vec4_base<unsigned int>;

using uchar2 = vec2_base<unsigned char>;
using uchar3 = vec3_base<unsigned char>;
using uchar4 = vec4_base<unsigned char>;

using ushort2 = vec2_base<unsigned short>;
using ushort3 = vec3_base<unsigned short>;
using ushort4 = vec4_base<unsigned short>;

using ulong2 = vec2_base<unsigned long>;
using ulong3 = vec3_base<unsigned long>;
using ulong4 = vec4_base<unsigned long>;

using long2 = vec2_base<long>;
using long3 = vec3_base<long>;
using long4 = vec4_base<long>;

using short2 = vec2_base<short>;
using short3 = vec3_base<short>;
using short4 = vec4_base<short>;

struct dim3 {
	unsigned int x = 1;
	unsigned int y = 1;
	unsigned int z = 1;
};