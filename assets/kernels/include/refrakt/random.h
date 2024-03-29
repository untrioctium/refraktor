__device__ constexpr static unsigned int rotl(const unsigned int x, int k) {
	return (x << k) | (x >> (32 - k));
}


template<typename Generator, typename FloatT>
struct rand_engine {

	__device__ vec2<FloatT> randgauss(FloatT std_dev = 1.0) {		
		FloatT r, sinang, cosang;
		sincospi( rand01() * 2, &sinang, &cosang);
		r = std_dev * (rand01() + rand01() + rand01() + rand01() - 2.0);
		return {r * cosang, r * sinang};
	}

	__device__ FloatT rand01() {
		Generator* const gen = static_cast<Generator*>(this);
		if constexpr (flamelib::is_same<FloatT, double>) {
			return static_cast<double>(gen->rand()) / 4'294'967'295.0;
		} else {
			return static_cast<float>(gen->rand()) / 4'294'967'295.0f;
		}
	}

	__device__ unsigned int randbit() {
		Generator* const gen = static_cast<Generator*>(this);
		return (gen->rand() >> 24) & 1;
	}

};

template<typename FloatT>
struct xoroshiro64 : public rand_engine<xoroshiro64<FloatT>, FloatT> {

	using u4 = unsigned int;

	u4 a; u4 b;

	__device__ u4 rand() {
		const u4 s0 = a;
		u4 s1 = b;
		const u4 result = s0 * 0x9E3779BB;

		s1 ^= s0;
		a = rotl(s0, 26) ^ s1 ^ (s1 << 9);
		b = rotl(s1, 13);

		return result;
	}
	
	__device__ void init(u4 seed) {
		a = 0xf1ea5eed;
		b = seed;
	}
};

template<typename FloatT>
struct jsf32 : public rand_engine<jsf32<FloatT>, FloatT> {

	using u4 = unsigned int;

	u4 a; u4 b; u4 c; u4 d;

	

};