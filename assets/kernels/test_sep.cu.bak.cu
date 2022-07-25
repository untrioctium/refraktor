#include <cooperative_groups.h>
//#include <cooperative_groups/memcpy_async.h>
using namespace cooperative_groups;

#include <refrakt/precision.h>
#include <refrakt/random.h>

struct particle {
	Real x;
	Real y;
	float color;
};

//#include <refrakt/hammersley.h>
#include <refrakt/flamelib.h>

#define DEBUG(...) if(this_grid().thread_rank() == 0) printf(__VA_ARGS__)

struct thread_state {
	unsigned char xid;
	unsigned short shuf;

	jsf32ctx rs;
	particle part;
	
	__device__ void new_shuf() { shuf = rs.rand() % NUM_SHUF_BUFS; }

};
__shared__ thread_state ts[THREADS_PER_BLOCK];
#define this_thread() ts[this_thread_block().thread_rank()]

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

__device__ void inflate_linear( unsigned int tss_id, const Real* fl, const Real* fr, const unsigned char* pl, const unsigned char* pr) {
	float sample_pos = tss_id / float(this_grid().group_dim().x);

	auto copy_and_interpolate = []( auto sample_pos, auto* ob, const auto* bl, const auto* br, auto bsize) {
		int per_thread = bsize / THREADS_PER_BLOCK + 1;
		for(int i = 0; i < per_thread; i++) {
			auto idx = i * THREADS_PER_BLOCK + this_thread_block().thread_rank();

			if(idx < bsize)
				ob[idx] = INTERP(bl[idx], br[idx], sample_pos);
		}
	};

	copy_and_interpolate( sample_pos, flame, fl, fr, FLAME_SIZE_REALS );
	if(pl != nullptr) copy_and_interpolate( sample_pos, palette_r, pl, pr, CHANNEL_SIZE );
	if(pl != nullptr) copy_and_interpolate( sample_pos, palette_g, pl + CHANNEL_SIZE, pr + CHANNEL_SIZE, CHANNEL_SIZE );
	if(pl != nullptr) copy_and_interpolate( sample_pos, palette_b, pl + CHANNEL_SIZE * 2, pr + CHANNEL_SIZE * 2, CHANNEL_SIZE );
}

__device__ __inline__
void load_new_shuffle_async(const unsigned short* const shufs, int pass_idx) {
	if(pass_idx % THREADS_PER_BLOCK == 0) {
		this_thread().new_shuf();
	}
	this_thread_block().sync();
	
	// queue up a load
	/*memcpy_async(this_thread_block(), 
		block_shuffle, 
		shufs + THREADS_PER_BLOCK * ts[pass_idx % THREADS_PER_BLOCK].shuf, 
		sizeof(unsigned short) * THREADS_PER_BLOCK);*/
}

__device__ unsigned long long xform_counters[NUM_XFORMS];
template<typename Warp>
__device__ 
Real flame_pass(Warp& warp, unsigned int pass_idx, const unsigned short* const shuf, particle& in, particle& out) {
	
		// every 32 passes, repopulate this warp's xid buffer
		if(pass_idx % 32 == 0) {
			this_thread().xid = select_xform(flame, this_thread().rs.rand_uniform());	
		}
		warp.sync();

		//DEBUG("xid: %d\n", ts[warp.meta_group_rank() * 32 + pass_idx % 32].xid);
		Real opacity = dispatch( 
			flame,
			ts[warp.meta_group_rank() * 32 + pass_idx % 32].xid, 
			in, out, &this_thread().rs
		);
		
		if(badvalue(out.x) || badvalue(out.y)) {
			out.x = this_thread().rs.rand_uniform() * 2.0 - 1.0;
			out.y = this_thread().rs.rand_uniform() * 2.0 - 1.0;
			opacity = 0.0;
		}
		
		//wait(this_thread_block());
		auto shuf_val = __ldg(&shuf[ts[pass_idx % THREADS_PER_BLOCK].shuf * THREADS_PER_BLOCK + this_thread_block().thread_rank()]);
		ts[shuf_val].part = out;
		//DEBUG("%f %f\n", out.x, out.y);
		load_new_shuffle_async(shuf, pass_idx + 1); // note: implicit block.sync()
		in = ts[this_thread_block().thread_rank()].part;
		return opacity;
}
/*
__device__ unsigned long long eff_block_hits;

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
estimate_efficiency( 
	const Real* const flame_params,
	const unsigned short* const shuf_bufs,
	const int warmup_count,
	const int screen_w,
	const int screen_h,
	double* efficiency) 
{
	auto warp = tiled_partition<32>(this_thread_block());
	auto tss_id = this_thread_block().group_index().x;
		
	this_thread().rs.init(0x8765309 + this_grid().thread_rank());
	inflate_linear(tss_id, flame_params, flame_params + FLAME_SIZE_REALS, nullptr, nullptr);
	load_new_shuffle_async(shuf_bufs, 0);

	particle loc_part[2];
	hammersley::sample(this_grid().thread_rank(), this_grid().size(), loc_part[0]);

	if( this_grid().thread_rank() == 0) eff_block_hits = 0;
	this_thread_block().sync();

	for( unsigned int i = 0; i < warmup_count; i++) {
		flame_pass(warp, i, shuf_bufs, loc_part[0], loc_part[1]);
	}

	particle transformed;
	if constexpr(HAS_FINAL_XFORM) {
		dispatch(
			flame,
			-1,
			loc_part[1], transformed, &this_thread().rs);
	} 
	else transformed = loc_part[1]; 
}*/

__device__ decltype(clock64()) global_start;
__device__ volatile bool should_bail;

__shared__ float2 ts_aa_offsets;

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
flame_step(
	const Real* const flame_params, 
	const unsigned char* const palettes, 
	const unsigned short* const shuf_bufs,
	const unsigned long long quality_target, const int warmup_count,
	const long long bailout, unsigned int seed,
	float4* const bins, const int screen_w, const int screen_h,
	unsigned long long* quality_counter, unsigned long long* pass_counter )
{
	if(this_grid().thread_rank() == 0) {
		should_bail = false;
		global_start = clock64();
		for(int i = 0; i < NUM_XFORMS; i++) xform_counters[i] = 0;
	}
	
	auto warp = tiled_partition<32>(this_thread_block());
	auto tss_id = this_thread_block().group_index().x;
		
	this_thread().rs.init(seed + this_grid().thread_rank());

	if(this_thread_block().thread_rank() == 0) {
		ts_aa_offsets = this_thread().rs.rand_gaussian(1/3.0);
	}
 
	particle loc_part[2];
	//hammersley::sample( (this_grid().thread_rank() + seed) % this_grid().size(), this_grid().size(), loc_part[0]);
	//float2 rg = this_thread().rs.rand_gaussian(.01);
	//loc_part[0].x += rg.x;
	//loc_part[0].y += rg.y;
	loc_part[0].x = this_thread().rs.rand_uniform() * 2.0 - 1.0;
	loc_part[0].y = this_thread().rs.rand_uniform() * 2.0 - 1.0;
	
	load_new_shuffle_async(shuf_bufs, 0);
	
	inflate_linear(tss_id, flame_params, flame_params + FLAME_SIZE_REALS, palettes, palettes + PALETTE_SIZE);
	this_thread_block().sync();
	
	for( unsigned int i = 0; i < warmup_count; i++) {
		flame_pass(warp, i, shuf_bufs, loc_part[0], loc_part[1]);
	}
	
	flamelib::sync_grid();
	for( unsigned int i = 0; 
		!should_bail; 
		i++ ) {
		Real opacity = flame_pass(warp, i + warmup_count, shuf_bufs, loc_part[0], loc_part[1]);
		
		particle transformed;
		if constexpr(HAS_FINAL_XFORM) {
			dispatch(
				flame,
				-1,
				loc_part[1], transformed, &this_thread().rs);
		} 
		else transformed = loc_part[1];
		
		Real tmp = flame[0] * transformed.x + flame[2] * transformed.y + flame[4];
		transformed.y = trunc(flame[1] * transformed.x + flame[3] * transformed.y + flame[5] + ts_aa_offsets.y);
		transformed.x = trunc(tmp + ts_aa_offsets.x);
		
		unsigned int hit = 0;
		if(transformed.x >= 0 && transformed.y >= 0 
		&& transformed.x < screen_w && transformed.y < screen_h 
		&& opacity > 0.0) {
			float4* bin = bins + int(transformed.y) * screen_w + int(transformed.x);
			unsigned char palette_idx = static_cast<unsigned char>( floor(transformed.color * 255.4999f) );
			bin->x += palette_r[palette_idx]/255.0f * opacity;
			bin->y += palette_g[palette_idx]/255.0f * opacity;
			bin->z += palette_b[palette_idx]/255.0f * opacity;
			bin->w += opacity;
			//DEBUG("adding (%f, %f, %f, %f)\n", palette_r[palette_idx] * opacity, palette_g[palette_idx] * opacity, palette_b[palette_idx] * opacity, opacity);
			hit = (unsigned int)(255.0f /* opacity*/);
		}
		hit = __reduce_add_sync(0xffffffff, hit);
		if(warp.thread_rank() == 0) {
			atomicAdd((unsigned long long*) quality_counter, hit);
		}

		//if(this_thread_block().thread_rank() == 0) block_pass_counters[tss_id]++;
		
		__threadfence();
		flamelib::sync_grid();
		if(this_grid().thread_rank() == 0) {
			should_bail = *quality_counter > quality_target || (clock64() - global_start) >= bailout || i > 100'000'000;
			(*pass_counter) += blockDim.x * gridDim.x;
		}
		__threadfence();
		flamelib::sync_grid();
	}

	/*if(this_grid().thread_rank() == 0) {
		unsigned long long total = 0;
		for(int i = 0; i < NUM_XFORMS; i++)
			total += xform_counters[i];

		for(int i = 0; i < NUM_XFORMS; i++)
			printf("xform %d: %f%%, expected %f%%\n", 
				i, 
				xform_counters[i] * 100.0f / total,
				*(flame + xform_offsets[i]) * 100.0f / flame[6]
			);
			printf("%llu clock cycles\n", clock64() - global_start);
	}*/
}