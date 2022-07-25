namespace flamelib {
	__device__ constexpr auto warp_size() { return 32; }

	__device__ inline void sync_warp() { __syncwarp(); }
	__device__ inline void sync_block() { __syncthreads(); }
	
	__device__ inline bool is_warp_leader() { return threadIdx.x % warp_size() == 0; }
	__device__ inline bool is_block_leader() { return threadIdx.x == 0; }
	__device__ inline bool is_grid_leader() { return threadIdx.x == 0 && blockIdx.x == 0; };
	
	__device__ inline auto warp_rank() { return threadIdx.x % warp_size(); }
	__device__ inline auto block_rank() { return threadIdx.x; }
	__device__ inline auto grid_rank() { return threadIdx.x + blockIdx.x * blockDim.x; }
	
	__device__ inline auto warp_start_in_block() { return block_rank() - warp_rank(); }
}