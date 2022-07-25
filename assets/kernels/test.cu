#ifdef DOUBLE_PRECISION
	using Real = double;
	using vec2 = double2;
	using vec3 = double3;
	using vec4 = double4;
	#define M_PI 3.14159265358979323846264338327950288419
	#define M_1_PI (1.0/3.14159265358979323846264338327950288419)
	#define M_EPS (1e-10)
	#define INTERP(a, b, mix) ((a) * (1.0 - (mix)) + (b) * (mix))
#else
	using Real = float;
	using vec2 = float2;
	using vec3 = float3;
	using vec4 = float4;
	#define M_PI 3.14159265358979323846264338327950288419f
	#define M_1_PI (1.0f/3.14159265358979323846264338327950288419f)
	#define M_EPS (1e-10)
	#define INTERP(a, b, mix) ((a) * (1.0f - (mix)) + (b) * (mix))
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
using namespace cooperative_groups;

#include <refrakt/random.h>
#include <refrakt/hammersley.h>

#define DEBUG(...) if(this_grid().thread_rank() == 0) printf(__VA_ARGS__)

struct thread_state {
	unsigned int xid;
	unsigned int shuf;

	jsf32ctx rs;
	vec3 part;
	
	__device__ void new_shuf() { shuf = rs.rand() % NUM_SHUF_BUFS; }

};
__shared__ thread_state ts[THREADS_PER_BLOCK];
#define this_thread() ts[this_thread_block().thread_rank()]

__shared__ Real flame[FLAME_REAL_COUNT];

#include "flame_generated.h"

#define CHANNEL_SIZE 256
#define PALETTE_SIZE 768
__shared__ unsigned char palette_r[CHANNEL_SIZE];
__shared__ unsigned char palette_g[CHANNEL_SIZE];
__shared__ unsigned char palette_b[CHANNEL_SIZE];
__shared__ unsigned short block_shuffle[THREADS_PER_BLOCK];

#define flame_idx_ss_affine 0
#define flame_idx_weight_sum 6


__device__ __inline__
void load_new_shuffle_async(const unsigned short* const shufs, int pass_idx) {
	if(pass_idx % THREADS_PER_BLOCK == 0) {
		this_thread().new_shuf();
	}
	this_thread_block().sync();
	
	// queue up a load
	memcpy_async(this_thread_block(), 
		block_shuffle, 
		shufs + THREADS_PER_BLOCK * ts[pass_idx % THREADS_PER_BLOCK].shuf, 
		sizeof(unsigned short) * THREADS_PER_BLOCK);
}

template<typename Warp>
__device__ 
Real flame_pass(Warp& warp, unsigned int pass_idx, const unsigned short* const shuf, vec3& in, vec3& out) {
	
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
		
		wait(this_thread_block());
		ts[block_shuffle[this_thread_block().thread_rank()]].part = out;
		//DEBUG("%f %f\n", out.x, out.y);
		load_new_shuffle_async(shuf, pass_idx + 1); // note: implicit block.sync()
		in = ts[this_thread_block().thread_rank()].part;
		return opacity;
}

__device__ void inflate( unsigned int tss_id, float time, float degrees_per_second, float tss_width ) {
	//DEBUG("%f\n", time);
	float sample_pos = (tss_id / float(this_grid().group_dim().x) - 0.5f) * tss_width + time;
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
	
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
flame_step(
	const Real* const flame_params, 
	const unsigned char* const palettes, 
	const unsigned short* const shuf_bufs,
	const float quality_target, const int warmup_count,
	const long long bailout, unsigned int seed,
	float4* const bins, const int screen_w, const int screen_h,
	unsigned long long* const quality_counter, unsigned long long* const pass_counter,
	const float time, const float degrees_per_second, const float tss_width)
{
	auto start = clock64();
	
	auto warp = tiled_partition<32>(this_thread_block());
	auto tss_id = this_thread_block().group_index().x;

	
	memcpy_async(this_thread_block(), flame, 
		flame_params /*+ tss_id * FLAME_SIZE_BYTES*/, FLAME_SIZE_BYTES);
		
	this_thread().rs.init(seed + this_grid().thread_rank());
	vec3 loc_part[2];
	hammersley::sample( this_grid().thread_rank(), this_grid().size(), loc_part[0]);
	//loc_part[0].x = this_thread().rs.rand_uniform() * 2.0 - 1.0;
	//loc_part[0].y = this_thread().rs.rand_uniform() * 2.0 - 1.0;
	loc_part[0].z = this_thread().rs.rand_uniform();
	
	wait(this_thread_block());
	load_new_shuffle_async(shuf_bufs, 0);
	memcpy_async(this_thread_block(), palette_r, palettes /*+ PALETTE_SIZE * tss_id*/                   , CHANNEL_SIZE);
	memcpy_async(this_thread_block(), palette_g, palettes /*+ PALETTE_SIZE * tss_id*/ + CHANNEL_SIZE    , CHANNEL_SIZE);
	memcpy_async(this_thread_block(), palette_b, palettes /*+ PALETTE_SIZE * tss_id*/ + CHANNEL_SIZE * 2, CHANNEL_SIZE);
	
	if(this_thread_block().thread_rank() == 0) inflate(tss_id, time, degrees_per_second, tss_width);
	this_thread_block().sync();
	
	for( unsigned int i = 0; i < warmup_count; i++) {
		flame_pass(warp, i, shuf_bufs, loc_part[0], loc_part[1]);
	}
	
	for( unsigned int i = 0; 
		float(*quality_counter)/255.0f < quality_target &&
		(clock64() - start) < bailout &&
		i < 100'000'000; 
		i++ ) {
		Real opacity = flame_pass(warp, i + warmup_count, shuf_bufs, loc_part[0], loc_part[1]);
		
		vec3 transformed;
		if constexpr(false) {
			dispatch(
				flame,
				-1,
				loc_part[1], transformed, &this_thread().rs);
				transformed.z = loc_part[1].z;
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
		if(warp.thread_rank() == 0) atomicAdd(quality_counter, hit);
		if(this_thread_block().thread_rank() == 0) atomicAdd(pass_counter, THREADS_PER_BLOCK);
		
		this_thread_block().sync();
	}
}