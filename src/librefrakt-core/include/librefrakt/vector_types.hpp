#pragma once

namespace rfkt {

	template<typename T>
	struct vec2 {
		T x, y;
	};

	template<typename T>
	struct vec3 {
		T x, y, z;
	};

	template<typename T>
	struct vec4 {
		T x, y, z, w;
	};

	using int2 = vec2<int>;
	using int3 = vec3<int>;
	using int4 = vec4<int>;

	using float2 = vec2<float>;
	using float3 = vec3<float>;
	using float4 = vec4<float>;

	using double2 = vec2<double>;
	using double3 = vec3<double>;
	using double4 = vec4<double>;

	using uint2 = vec2<unsigned int>;
	using uint3 = vec3<unsigned int>;
	using uint4 = vec4<unsigned int>;

	using uchar2 = vec2<unsigned char>;
	using uchar3 = vec3<unsigned char>;
	using uchar4 = vec4<unsigned char>;

	using ushort2 = vec2<unsigned short>;
	using ushort3 = vec3<unsigned short>;
	using ushort4 = vec4<unsigned short>;

	using half3 = ushort3;
	using half4 = ushort4;

	using ulong2 = vec2<unsigned long>;
	using ulong3 = vec3<unsigned long>;
	using ulong4 = vec4<unsigned long>;

	using long2 = vec2<long>;
	using long3 = vec3<long>;
	using long4 = vec4<long>;

	using short2 = vec2<short>;
	using short3 = vec3<short>;
	using short4 = vec4<short>;

}