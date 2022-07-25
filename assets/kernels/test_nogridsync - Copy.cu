#include <refrakt/precision.h>
#include <refrakt/random.h>

struct particle {
	Real x;
	Real y;
	float color;
};

#include <refrakt/flamelib.h>
#include <cooperative_groups/memcpy_async.h>

namespace fl = flamelib;
#include <refrakt/hammersley.h>


#define DEBUG(...) if(this_grid().thread_rank() == 0) printf(__VA_ARGS__)

#define CHANNEL_SIZE 256
#define PALETTE_SIZE 768

struct thread_state {
	unsigned char xid;
	unsigned short shuf;

	jsf32ctx rs;
	particle part;
	
	__device__ void new_shuf() { shuf = rs.rand() % NUM_SHUF_BUFS; }

};

struct shared_state {

  thread_state ts[THREADS_PER_BLOCK];
  unsigned char palette[PALETTE_SIZE];
  unsigned short block_shuffle[THREADS_PER_BLOCK];
  float2 ts_aa_offsets;
  
  Real flame[FLAME_SIZE_REALS];

  __device__ unsigned char* palette_r() { return palette; }
  __device__ unsigned char* palette_g() { return palette + 256; }
  __device__ unsigned char* palette_b() { return palette + 512; }
};

__shared__ shared_state block_state;

#define this_thread() block_state.ts[fl::block_rank()]


#define flame_idx_ss_affine 0
#define flame_idx_weight_sum 6

#include "flame_generated.h"

__device__ void inflate_linear( unsigned int tss_id, const Real* fl, const Real* fr, const unsigned char* pl, const unsigned char* pr) {
	float sample_pos = tss_id / float(gridDim.x);

	auto copy_and_interpolate = []( auto sample_pos, auto* ob, const auto* bl, const auto* br, auto bsize) {
		int per_thread = bsize / THREADS_PER_BLOCK + 1;
		for(int i = 0; i < per_thread; i++) {
			auto idx = i * THREADS_PER_BLOCK + fl::block_rank();

			if(idx < bsize)
				ob[idx] = INTERP(bl[idx], br[idx], sample_pos);
		}
	};

	copy_and_interpolate( sample_pos, block_state.flame, fl, fr, FLAME_SIZE_REALS );
	if(pl != nullptr) copy_and_interpolate( sample_pos, block_state.palette, pl, pr, PALETTE_SIZE );
}

__device__ __inline__
void load_new_shuffle_async(const unsigned short* const shufs, int pass_idx) {
	if(pass_idx % THREADS_PER_BLOCK == 0) {
		this_thread().new_shuf();
	}
	fl::sync_block();
	
	// queue up a load
	cooperative_groups::memcpy_async(cooperative_groups::this_thread_block(), 
		block_state.block_shuffle, 
		shufs + THREADS_PER_BLOCK * block_state.ts[pass_idx % THREADS_PER_BLOCK].shuf, 
		sizeof(unsigned short) * THREADS_PER_BLOCK);
}

__device__ 
Real flame_pass(unsigned int pass_idx, const unsigned short* const shuf, particle& in, particle& out) {
	
		// every 32 passes, repopulate this warp's xid buffer
		if(pass_idx % 32 == 0) {
			this_thread().xid = select_xform(block_state.flame, this_thread().rs.rand_uniform());	
		}
		fl::sync_warp();

		//DEBUG("xid: %d\n", ts[warp.meta_group_rank() * 32 + pass_idx % 32].xid);
		Real opacity = dispatch( 
			block_state.flame,
			block_state.ts[fl::warp_start_in_block() + pass_idx % 32].xid, 
			in, out, &this_thread().rs
		);
		
		/*if(badvalue(out.x) || badvalue(out.y)) {
			out.x = this_thread().rs.rand_uniform() * 2.0 - 1.0;
			out.y = this_thread().rs.rand_uniform() * 2.0 - 1.0;
			opacity = 0.0;
		}*/
		
		cooperative_groups::wait(cooperative_groups::this_thread_block());
		auto shuf_val = block_state.block_shuffle[fl::block_rank()];
		block_state.ts[shuf_val].part = out;
		//DEBUG("%f %f\n", out.x, out.y);
		load_new_shuffle_async(shuf, pass_idx + 1); // note: implicit block.sync()
		in = block_state.ts[fl::block_rank()].part;
		return opacity;
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
flame_step(
	const Real* const flame_params, 
	const unsigned char* const palettes, 
	const unsigned short* const shuf_bufs,
	const unsigned long long quality_target, const int warmup_count, const int iter_bailout,
	const long long bailout, unsigned int seed,
	float4* const bins, const int screen_w, const int screen_h,
	unsigned long long* quality_counter, unsigned long long* pass_counter )
{
	
	__shared__ unsigned long long tss_quality;
	__shared__ unsigned long long tss_passes;
	__shared__ decltype(clock64()) tss_start;
	__shared__ bool should_bail;
	
	auto tss_id = blockIdx.x;
	this_thread().rs.init(seed + fl::grid_rank());

	if(fl::is_block_leader()) {
		block_state.ts_aa_offsets = this_thread().rs.rand_gaussian(1/3.0);
		tss_quality = 0;
		tss_passes = 0;
		should_bail = false;
	}
 
	particle loc_part[2];
	hammersley::sample( (fl::grid_rank() + seed) % gridDim.x, gridDim.x, loc_part[0]);
	float2 rg = this_thread().rs.rand_gaussian(.01);
	loc_part[0].x += rg.x;
	loc_part[0].y += rg.y;
	loc_part[0].color = this_thread().rs.rand_uniform();
	
	load_new_shuffle_async(shuf_bufs, 0);
	
	inflate_linear(tss_id, flame_params, flame_params + FLAME_SIZE_REALS, palettes, palettes + PALETTE_SIZE);
	fl::sync_block();
	
	for( unsigned int i = 0; i < warmup_count; i++) {
		flame_pass(i, shuf_bufs, loc_part[0], loc_part[1]);
	}
	
	fl::sync_block();
	if(fl::is_block_leader()) tss_start = clock64();
	fl::sync_block();
	for( unsigned int i = 0; 
		!should_bail; 
		i++ ) {
		Real opacity = flame_pass(i + warmup_count, shuf_bufs, loc_part[0], loc_part[1]);
		
		particle transformed;
		if constexpr(HAS_FINAL_XFORM) {
			dispatch(
				block_state.flame,
				-1,
				loc_part[1], transformed, &this_thread().rs);
		} 
		else transformed = loc_part[1];
		
		Real tmp = block_state.flame[0] * transformed.x + block_state.flame[2] * transformed.y + block_state.flame[4];
		transformed.y = trunc(block_state.flame[1] * transformed.x + block_state.flame[3] * transformed.y + block_state.flame[5] + block_state.ts_aa_offsets.y);
		transformed.x = trunc(tmp + block_state.ts_aa_offsets.x);
		
		unsigned int hit = 0;
		if(transformed.x >= 0 && transformed.y >= 0 
		&& transformed.x < screen_w && transformed.y < screen_h 
		&& opacity > 0.0) {
			float4* bin = bins + int(transformed.y) * screen_w + int(transformed.x);
			unsigned char palette_idx = static_cast<unsigned char>( floor(transformed.color * 255.4999f) );
			bin->x += block_state.palette_r()[palette_idx]/255.0f * opacity;
			bin->y += block_state.palette_g()[palette_idx]/255.0f * opacity;
			bin->z += block_state.palette_b()[palette_idx]/255.0f * opacity;
			bin->w += opacity;
			//DEBUG("adding (%f, %f, %f, %f)\n", palette_r[palette_idx] * opacity, palette_g[palette_idx] * opacity, palette_b[palette_idx] * opacity, opacity);
			hit = (unsigned int)(255.0f * opacity);
		}
		hit = __reduce_add_sync(0xffffffff, hit);
		if(fl::is_warp_leader()) {
			atomicAdd(&tss_quality, hit);
		}

		//if(this_thread_block().thread_rank() == 0) block_pass_counters[tss_id]++;
		
		//__threadfence();
		fl::sync_block();
		if(fl::is_block_leader()) {
			should_bail = tss_quality > (quality_target / gridDim.x) || (clock64() - tss_start) >= bailout || i >= iter_bailout;
			tss_passes += blockDim.x;
		}
		//__threadfence();
		fl::sync_block();
	}
	
	if(fl::is_block_leader()) {
		atomicAdd(quality_counter, tss_quality);
		atomicAdd(pass_counter, tss_passes);
	}
}