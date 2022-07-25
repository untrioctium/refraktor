namespace hammersley {
	
	namespace detail {
		
		constexpr __device__ unsigned int ReverseBits32(unsigned int n) {
			n = (n << 16) | (n >> 16);
			n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
			n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
			n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
			n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
			return n;
		}
		
		constexpr __device__ unsigned int flip_bits(unsigned int v, unsigned int radix) {
			return ReverseBits32(v) >> (32 - radix);
		}

		constexpr __device__ unsigned int next_pow2(unsigned int v) {
			v--;
			v |= v >> 1;
			v |= v >> 2;
			v |= v >> 4;
			v |= v >> 8;
			v |= v >> 16;
			v++;
			return v;
		}
	}
	
	__device__ vec2 sample( unsigned int idx, unsigned int total ) {
		Real inv_max;
		
		unsigned radix = 0;
		{
			unsigned int max_val = (total % 2 == 0) ? total: detail::next_pow2(total);
			unsigned int v = max_val;
			while(v >>= 1) radix++;
			
			inv_max = static_cast<Real>(1.0)/max_val;
		}
		
		return {
			idx * inv_max * static_cast<Real>(2.0) - static_cast<Real>(1.0),
			detail::flip_bits(idx, radix) * inv_max * static_cast<Real>(2.0) - static_cast<Real>(1.0)
		};
	}
}