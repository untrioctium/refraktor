#include <refrakt/precision.h>
#include <refrakt/random.h>
#include <refrakt/hammersley.h>
#include <refrakt/flamelib.h>

namespace fl = flamelib;

#define DEBUG(...) if(this_grid().thread_rank() == 0) printf(__VA_ARGS__)

struct thread_state {
	unsigned int xid;
	unsigned int shuf;

	jsf32ctx rs;
	vec3 part;
	
	__device__ void new_shuf() { shuf = rs.rand() % NUM_SHUF_BUFS; }

};
__shared__ thread_state ts[THREADS_PER_BLOCK];
#define this_thread() ts[fl::block_rank()]

__shared__ Real flame[FLAME_SIZE_REALS];

#define CHANNEL_SIZE 256
#define PALETTE_SIZE 768
__shared__ unsigned char palette_r[CHANNEL_SIZE];
__shared__ unsigned char palette_g[CHANNEL_SIZE];
__shared__ unsigned char palette_b[CHANNEL_SIZE];
__shared__ unsigned short block_shuffle[THREADS_PER_BLOCK];

#define flame_idx_ss_affine 0
#define flame_idx_weight_sum 6

#include "flame_generated.h"

__device__ inline Real clamp( Real val, Real min_v, Real max_v ) {
	return max( min_v, min( max_v, val ));
}

__device__ inline Real motion_sin( Real time ) {
	return sin( time * 2.0 * M_PI );
}

__device__ inline Real motion_triangle( Real time ) {
	Real fr = fmod(time, 1.0f);
	if(fr < 0) fr += 1.0f;
	
	if( fr <= .25f )
		return 4.0f * fr;
	if( fr <= .75 )
		return -4.0f * fr + 2.0;
	
	return 4.0f * fr - 4.0f;
}

__device__ void inflate( unsigned int tss_id, float time, float degrees_per_second, float tss_width ) {
	//DEBUG("%f\n", time);
	float sample_pos = (tss_id / float(blockDim.x) - 0.5f) * tss_width + time;
	for(int i = 0; i < NUM_XFORMS; i++) {
		Real* xf = flame + xform_offsets[i];
		
		Real so, co;
		sincos(0.017453292519943295769236907684f * degrees_per_second * xf[4] * sample_pos, &so, &co);
		
		Real old = xf[5];
		xf[5] = xf[5] * co + xf[7] * so;
		xf[7] = xf[7] * co - old * so;
		old = xf[6];
		xf[6] = xf[6] * co + xf[8] * so;
		xf[8] = xf[8] * co - old * so;	
	}
	
	//flame[8] += 0.75 * motion_sin(sample_pos/5.0f);
	//flame[26] += 0.75 * motion_sin(sample_pos/5.0f);
	//flame[44] += 0.75 * motion_sin(sample_pos/5.0f);
}

__device__ Real dispatch( Real* flame, int idx, vec3& in, vec3& out, jsf32ctx* rs );
__device__ unsigned int select_xform(Real* flame, Real ratio);
__device__ unsigned int flame_size_reals();
__device__ unsigned int flame_size_bytes();

__device__ __inline__
void load_new_shuffle_async(const unsigned short* const shufs, int pass_idx) {
	if(pass_idx % THREADS_PER_BLOCK == 0) {
		this_thread().new_shuf();
	}
	fl::sync_block();
	
	// queue up a load
	/*cg::memcpy_async( cg::this_thread_block(),
		block_shuffle, 
		shufs + THREADS_PER_BLOCK * ts[pass_idx % THREADS_PER_BLOCK].shuf, 
		sizeof(unsigned short) * THREADS_PER_BLOCK);*/
	
	block_shuffle[fl::block_rank()] = shufs[THREADS_PER_BLOCK * ts[pass_idx % THREADS_PER_BLOCK].shuf + fl::block_rank()];
	fl::sync_block();
}

__device__ 
Real flame_pass(unsigned int pass_idx, const unsigned short* const shuf, vec3& in, vec3& out) {
	
		// every 32 passes, repopulate this warp's xid buffer
		if(pass_idx % 32 == 0) {
			this_thread().xid = select_xform(flame, this_thread().rs.rand_uniform());	
		}
		fl::sync_warp();
		//DEBUG("xid: %d\n", ts[warp.meta_group_rank() * 32 + pass_idx % 32].xid);
		Real opacity = dispatch( 
			flame,
			ts[fl::warp_start_in_block() + pass_idx % 32].xid, 
			in, out, &this_thread().rs
		);
		
		fl::await_memcpy();
		ts[block_shuffle[fl::block_rank()]].part = out;
		//DEBUG("%f %f\n", out.x, out.y);
		load_new_shuffle_async(shuf, pass_idx + 1); // note: implicit block.sync()
		in = ts[fl::block_rank()].part;
		return opacity;
}

__device__ decltype(clock64()) global_start;
__device__ decltype(clock64()) global_duration;

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
flame_step(
	const Real* const flame_params, 
	const unsigned char* const palettes, 
	const unsigned short* const shuf_bufs,
	const unsigned long long quality_target, const int warmup_count,
	const long long bailout, unsigned int seed,
	float4* const bins, const int screen_w, const int screen_h,
	unsigned long long* const quality_counter, unsigned long long* const block_pass_counters,
	const float time, const float degrees_per_second, const float tss_width)
{
	if(fl::is_grid_leader()) global_start = clock64();
	
	auto tss_id = blockIdx.x;

	fl::memcpy_async( flame, 
		flame_params /*+ tss_id * FLAME_SIZE_BYTES*/, flame_size_bytes());
		
	this_thread().rs.init(seed + fl::grid_rank());
	vec3 loc_part[2];
	//hammersley::sample( this_grid().thread_rank(), this_grid().size(), loc_part[0]);
	loc_part[0].x = this_thread().rs.rand_uniform() * 2.0 - 1.0;
	loc_part[0].y = this_thread().rs.rand_uniform() * 2.0 - 1.0;
	loc_part[0].z = this_thread().rs.rand_uniform();
	
	fl::await_memcpy();
	load_new_shuffle_async(shuf_bufs, 0);
	fl::memcpy_async(palette_r, palettes /*+ PALETTE_SIZE * tss_id*/                   , CHANNEL_SIZE);
	fl::memcpy_async(palette_g, palettes /*+ PALETTE_SIZE * tss_id*/ + CHANNEL_SIZE    , CHANNEL_SIZE);
	fl::memcpy_async(palette_b, palettes /*+ PALETTE_SIZE * tss_id*/ + CHANNEL_SIZE * 2, CHANNEL_SIZE);
	
	if(fl::is_block_leader()) inflate(tss_id, time, degrees_per_second, tss_width);
	fl::sync_block();
	
	for( unsigned int i = 0; i < warmup_count; i++) {
		flame_pass(i, shuf_bufs, loc_part[0], loc_part[1]);
	}
	
	fl::sync_grid();
	if(fl::is_grid_leader()) global_duration = clock64() - global_start;
	fl::sync_grid();
	
	for( unsigned int i = 0; 
		*quality_counter < quality_target &&
		global_duration < bailout &&
		i < 100'000'000; 
		i++ ) {
		Real opacity = flame_pass(i + warmup_count, shuf_bufs, loc_part[0], loc_part[1]);
		
		vec3 transformed;
		if constexpr(HAS_FINAL_XFORM) {
			dispatch(
				flame,
				-1,
				loc_part[1], transformed, &this_thread().rs);
				//transformed.z = loc_part[1].z;
		} 
		else transformed = loc_part[1];
		
		Real tmp = flame[0] * transformed.x + flame[2] * transformed.y + flame[4];
		transformed.y = flame[1] * transformed.x + flame[3] * transformed.y + flame[5];
		transformed.x = tmp;
		
		unsigned int hit = 0;
		if(transformed.x >= 0 && transformed.y >= 0 
		&& transformed.x < screen_w && transformed.y < screen_h 
		&& opacity > 0.0) {
			float4* bin = bins + int(transformed.y) * screen_w + int(transformed.x);
			unsigned char palette_idx = static_cast<unsigned char>( floor(transformed.z * 255.4999f) );
			bin->x += palette_r[palette_idx]/255.0f * opacity;
			bin->y += palette_g[palette_idx]/255.0f * opacity;
			bin->z += palette_b[palette_idx]/255.0f * opacity;
			bin->w += opacity;
			//DEBUG("adding (%f, %f, %f, %f)\n", palette_r[palette_idx] * opacity, palette_g[palette_idx] * opacity, palette_b[palette_idx] * opacity, opacity);
			hit = (unsigned int)(255.0f * opacity);
		}
		hit = __reduce_add_sync(0xffffffff, hit);
		if(fl::is_warp_leader()) {
			atomicAdd(quality_counter, hit);
		}

		//if(this_thread_block().thread_rank() == 0) block_pass_counters[tss_id]++;
		
		fl::sync_grid();
		if(fl::is_grid_leader()) global_duration = clock64() - global_start;
		fl::sync_grid();
	}
}