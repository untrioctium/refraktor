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
}