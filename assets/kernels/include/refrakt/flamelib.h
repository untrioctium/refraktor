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
		void* sort_storage;
		unsigned long long* samples;
		unsigned long long tss_quality;
		unsigned long long tss_passes;
		uint64 tss_start;
		bool should_bail;

		FlameT flame;
	};
}