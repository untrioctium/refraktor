struct jsf32ctx {

	using u4 = unsigned int;

	u4 a; u4 b; u4 c; u4 d;

	#define rot32(x,k) (((x)<<(k))|((x)>>(32-(k))))
	__device__ __host__ u4 rand() {
		u4 e = a - rot32(b, 27);
		a = b ^ rot32(c, 17);
		b = c + d;
		c = d + e;
		d = e + a;
		return d;
	}
	#undef rot32
	
	__device__ void init(u4 seed) {
		a = 0xf1ea5eed;
		b = c = d = seed;

		#pragma unroll
		for (int i = 0; i < 20; i++) (void) this->rand();
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