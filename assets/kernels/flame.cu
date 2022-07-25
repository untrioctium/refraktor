#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using Real = float;
using vec2 = float2;
using vec3 = float3;
using vec4 = float4;

#include "random.hpp"
#include "hammersley.hpp"
#include "math_helpers.hpp"

__constant__ int constants[num_constants];

__shared__ Real  flame[flame_reals];
__shared__ unsigned char palette_r[256];
__shared__ unsigned char palette_g[256];
__shared__ unsigned char palette_b[256];
__shared__ unsigned char xid[warps_per_block * threads_per_warp];
__shared__ vec3 particle_buffer[threads_per_block];
__shared__ jsf32ctx rand_states[threads_per_block];
__shared__ unsigned short block_shuffle[threads_per_block]
__shared__ unsigned char block_shuffle_idx[warps_per_block];

#define thread_rand rand_states[block.thread_rank()]

#include "flame_generated.hpp"

// universal constants
constexpr unsigned int palette_size = sizeof(unsigned char) * 3 * 256;

// program constants
constexpr unsigned char num_xforms = 4;
constexpr bool has_final_xform = false;
constexpr unsigned int flame_size_bytes = 400;
constexpr unsigned int xform_offsets[] = {1, 5, 6, 4};

#define flame_idx_ss_affine 0
#define flame_idx_weight_sum 6
#define flame_idx_weights 7

namespace cg = cooperative_groups;
namespace refrakt::kernel {
	enum class constants: char {
		screen_w,
		screen_h
		num_constants
	}
}

__device__ __inline__ 
unsigned char select_xform(Real ratio) {
	ratio *= flame[flame_idx_weight_sum];
	
	#pragma unroll
	for(unsigned char i = 0; i < num_xforms; i++)
		if( *(flame + flame_idx_weights + i) >= ratio) return i;
		
	return num_xforms - 1;
}

template<typename Group>
__device__ __inline__
void load_new_shuffle_async(Group& block, unsigned int pass_idx) {
	// block leader selects the shuffle index
	if(block.thread_rank() == 0) block_shuffle_idx = rand_states[block.thread_rank()].rand() % constants[shuffle_buffer_count];
	block.sync();
	
	// queue up a load
	cg::memcpy_async(block, block_shuffle, block_shuffle + particles_per_block * block_shuffle_idx, sizeof(unsigned short) * particles_per_block);
}

template<typename Warp, Block>
__device__ 
void flame_pass(Block& block, Warp& warp, unsigned int pass_idx) {
	
		// every 32 passes, repopulate this warp's xid buffer
		if(pass_idx % threads_per_warp == 0) {
			xid[warp.meta_group_rank() * warp.size() + warp.thread_rank()] = select_xform(rand_states[block.thread_rank()].rand_uniform());	
		}
		warp.sync();
		
		flame_step( 
			xid[warp.meta_group_rank() * threads_per_warp + pass_idx % threads_per_warp], 
			loc_particle[0], loc_particle[1]
		);
		
		cg::wait(block);
		particle_buffer[block_shuffle[block.thread_rank()]] = loc_particle[1];
		load_new_shuffle_async(block); // note: implicit block.sync()
		loc_particle[0] = particle_buffer[block.thread_rank()];
}

__global__ void 
__launch_bounds__( threads_per_block, blocks_per_multiprocessor )
flame_kernel(
	Real* flame_params,

) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<warp_size>(block);
	
	auto tss_id = block.group_index().y;
	
	// initial loads
	cg::memcpy_async(block, flame, flame_params + tss_id * flame_size_bytes_aligned, flame_size_bytes_aligned);
	
	// warm thread rand state
	rand_states[block.thread_rank()].init( 0x8675309 + grid.thread_rank() );
	
	// pick a starting location
	vec3 loc_particle[2];
	hammersley::sample( grid.thread_rank(), grid.size(), loc_particle[0]);
	loc_particle[0].z = rand_states[block.thread_rank()].rand_uniform();
	
	cg::wait(block);
	load_new_shuffle_async(block);
	
	cg::memcpy_async(block, palette_r, palettes + palette_size * tss_id      , 256);
	cg::memcpy_async(block, palette_g, palettes + palette_size * tss_id + 256, 256);
	cg::memcpy_async(block, palette_b, palettes + palette_size * tss_id + 512, 256);
	
	for( unsigned int i = 0; i < constants[warmup_count]; i++) {
		flame_pass(block, warp, i);
	}
	
	for( unsigned int i = 0; i < constants[drawing_count]; i++ ) {
		flame_pass(block, warp, i + constants[warmup_count]);
		
		vec3 transformed;
		
		if constexpr(has_final_xform) {
			flame_step(
				-1, flame + xform_offsets[num_xforms],
				loc_particle[1], transformed);
		} 
		else transformed = loc_particle[1];
		
		transformed = apply_affine(transformed, flame + flame_idx_ss_affine);
		
		if(transformed.x >= 0 && transformed.y >= 0 && transformed.x < constants[screen_w] && transformed.y < constants[screen_h] && warp_xf_ptr[xform_idx_opacity] > 0.0) {
			vec4* bin = bins + transformed.y * constants[screen_w] + transformed.x;
			Real opacity = warp_xf_ptr[xform_idx_opacity];
			unsigned char palette_idx = static_cast<unsigned char>( transformed.z * 255.0f );
			atomicAdd( &bin->x, palette_r[palette_idx] * opacity);
			atomicAdd( &bin->y, palette_r[palette_idx] * opacity);
			atomicAdd( &bin->z, palette_r[palette_idx] * opacity);
			atomicAdd( &bin->w, opacity);
		}
	}
}