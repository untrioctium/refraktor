struct randctx {

	using u4 = unsigned int;

	u4 a; u4 b;

	#define rot32(x,k) (((x)<<(k))|((x)>>(32-(k))))
	__device__ __host__ u4 rand() {
		const u4 s0 = a;
		u4 s1 = b;
		const u4 result = s0 * 0x9E3779BB;

		s1 ^= s0;
		a = rot32(s0, 26) ^ s1 ^ (s1 << 9);
		b = rot32(s1, 13);

		return result;
	}
	#undef rot32
	
	__device__ void init(u4 seed) {
		a = 0xf1ea5eed;
		b = seed;
	}

	__device__ Real rand_uniform() {
		#ifdef DOUBLE_PRECISION
			unsigned long long bits = ((((unsigned long long) rand()) << 32) | (((unsigned long long) rand())));
			bits &= 0x000FFFFFFFFFFFFFull;
			bits |= 0x3FF0000000000000ull;
			return *(double*)(&bits) - 1.0;
		#else
			return static_cast<float>(rand()) / 4'294'967'295.0f;
		#endif
	}

	__device__ float2 rand_gaussian(float stddev = 1.0f) {
        Real r, sinang,cosang;
        sincos( rand_uniform() * 2 * M_PI, &sinang, &cosang );
        r = stddev * (rand_uniform() + rand_uniform() + rand_uniform() + rand_uniform() - 2.0);
		return { r * cosang, r * sinang };
	}
};