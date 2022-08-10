#include <cooperative_groups/memcpy_async.h>

#define DEBUG(...) if(threadIdx.x == 0 && blockIdx.x == 0) {printf(__VA_ARGS__);}

#include <refrakt/precision.h>
#include <refrakt/random.h>
#include <refrakt/flamelib.h>
#include <refrakt/hammersley.h>
#include <refrakt/color.h>

namespace fl = flamelib;
namespace cg = cooperative_groups;

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
constexpr static uint32 threads_per_block = THREADS_PER_BLOCK;
constexpr static uint32 flame_size_reals = FLAME_SIZE_REALS;
constexpr static uint32 flame_size_bytes = flame_size_reals * sizeof(Real);
constexpr static uint32 num_shuf_bufs = NUM_SHUF_BUFS;
constexpr static uint32 shuf_buf_size = threads_per_block * sizeof(uint16);

using iterator = float3;
struct thread_states_t {
	iterator iterators[threads_per_block];
	randctx rand_states[threads_per_block];
	uint16 shuffle_vote[threads_per_block];
	uint16 shuffle[threads_per_block];
	uint8 xform_vote[threads_per_block];
};

struct __align__(16) shared_state_t {
	thread_states_t ts;
	
	Real flame[flame_size_reals];
	uchar3 palette[palette_channel_size];
	
	float2 antialiasing_offsets;
	unsigned long long tss_quality;
	unsigned long long tss_passes;
	decltype(clock64()) tss_start;
	bool should_bail;
};

constexpr auto shared_size_bytes = sizeof(shared_state_t);

struct segment {
	Real a, b, c, d;
	
	auto sample(Real t) const { return a * t * t * t + b * t * t + c * t + d; }
};

__shared__ shared_state_t state;

#define my_iter() (state.ts.iterators[fl::block_rank()])
#define my_rand() (state.ts.rand_states[fl::block_rank()])
#define my_shuffle() (state.ts.shuffle[fl::block_rank()])
#define my_shuffle_vote() (state.ts.shuffle_vote[fl::block_rank()])
#define my_xform_vote() (state.ts.xform_vote[fl::block_rank()])

#include "flame_generated.h"

__global__
void print_debug_info() {
	#define output_sizeof(T) printf("sizeof(" #T ")=%llu (%.2f%% of total shared)\n", sizeof(T), sizeof(T) / float(sizeof(shared_state_t)) * 100.0f);
	
	output_sizeof(shared_state_t);
	output_sizeof(shared_state_t::flame);
	output_sizeof(shared_state_t::palette);
	printf("------\n");
	output_sizeof(thread_states_t);
	output_sizeof(thread_states_t::iterators);
	output_sizeof(thread_states_t::rand_states);
	output_sizeof(thread_states_t::shuffle_vote);
	output_sizeof(thread_states_t::shuffle);
	output_sizeof(thread_states_t::xform_vote);
	printf("------\n");
	output_sizeof(iterator);

	printf("bytes per iterator: %llu (%llu leftover)\n", sizeof(thread_states_t)/THREADS_PER_BLOCK, sizeof(thread_states_t) % THREADS_PER_BLOCK);
}

__device__ 
void queue_shuffle_load(const uint16* const shufs, uint32 pass_idx) {
	if(pass_idx % threads_per_block == 0) {
		my_shuffle_vote() = my_rand().rand() % num_shuf_bufs;
	}
	
	fl::sync_block();
	cg::memcpy_async(cg::this_thread_block(), 
		state.ts.shuffle, 
		shufs + threads_per_block * state.ts.shuffle_vote[pass_idx % threads_per_block], 
		shuf_buf_size);
	
}

template<uint32 count>
void interpolate_flame( Real t, const segment* const __restrict__ seg, Real* const __restrict__ out ) {
	constexpr static uint32 per_thread = count / threads_per_block + ((count % threads_per_block)? 1: 0);
	//printf("(%d,%d,%d) ", count / threads_per_block, count % threads_per_block, per_thread);
	for(uint32 i = 0; i < per_thread; i++) {
		const auto idx = i * threads_per_block + fl::block_rank();
		
		if(idx < count) {
			out[idx] = seg[idx].sample(t);
		}
	}
	
}

template<uint32 count>
void interpolate_palette( Real t, const segment* const __restrict__ seg, uchar3* const __restrict__ palette )
{
	constexpr static uint32 per_thread = count / threads_per_block + ((count % threads_per_block)? 1: 0);

	for(uint32 i = 0; i < per_thread; i++) {
		const auto idx = i * threads_per_block + fl::block_rank();
		
		if(idx < count) {
			palette[idx] = 
				hsv_to_rgb(
					seg[3 * idx].sample(t),
					seg[3 * idx + 1].sample(t),
					seg[3 * idx + 2].sample(t)
				);
		}
	}
	
}

__device__ 
Real flame_pass(unsigned int pass_idx, const unsigned short* const shuf) {
	
	// every 32 passes, repopulate this warp's xid buffer
	if(pass_idx % 32 == 0) {
		my_xform_vote() = select_xform(my_rand().rand_uniform());
	}
	fl::sync_warp();

	auto out_local = iterator{-666.0, -666.0, -660.0};
	//DEBUG("xid: %d\n", ts[warp.meta_group_rank() * 32 + pass_idx % 32].xid);
	Real opacity = dispatch( 
		state.ts.xform_vote[fl::warp_start_in_block() + pass_idx % 32], 
		my_iter(), out_local
	);
	//out_local = iterator{{-666.0, -666.0}, -660.0};

	if(badvalue(out_local.x) || badvalue(out_local.y)) {
		out_local.x = my_rand().rand_uniform() * 2.0 - 1.0;
		out_local.y = my_rand().rand_uniform() * 2.0 - 1.0;
		opacity = 0.0;
	}
	
	cg::wait(cg::this_thread_block());
	state.ts.iterators[my_shuffle()] = out_local;
	queue_shuffle_load(shuf, pass_idx + 1); // note: implicit block.sync()
	return opacity;
}


__global__ 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
void warmup(
	const uint32 num_segments,
	const segment* const __restrict__ segments, 
	const uint16* const __restrict__ shuf_bufs,
	const uint32 seed, const uint32 warmup_count,
	shared_state_t* __restrict__ out_state ) 
{	
	my_rand().init(seed + fl::grid_rank());
	
	my_iter() = {my_rand().rand_uniform(), my_rand().rand_uniform(), my_rand().rand_uniform()};//hammersley::sample( fl::grid_rank(), gridDim.x );
	
	if(fl::is_block_leader()) {
		state.antialiasing_offsets = my_rand().rand_gaussian(1/3.0);
	}
	queue_shuffle_load(shuf_bufs, 0);
	
	const auto t = blockIdx.x/Real(gridDim.x) * num_segments;
	const auto seg_size = gridDim.x / num_segments;
	const auto seg = blockIdx.x / seg_size;
	const auto seg_offset = flame_size_reals + palette_size;
	
	interpolate_flame<flame_size_reals>(t, segments + (seg_offset) * seg, state.flame);
	interpolate_palette<256>(t, segments + flame_size_reals + (seg_offset) * seg, state.palette);
	
	fl::sync_block();
	for(unsigned int pass = 0; pass < warmup_count; pass++) {
		flame_pass(pass, shuf_bufs);
	}
	
	// save shared state
	cg::memcpy_async(cg::this_thread_block(), out_state + blockIdx.x, &state, shared_size_bytes);
	cg::wait(cg::this_thread_block());
}


__global__
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
void bin(
	shared_state_t* const __restrict__ in_state,
	const uint16* const __restrict__ shuf_bufs,
	const uint64 quality_target,
	const uint32 iter_bailout,
	const uint64 time_bailout,
	float4* const __restrict__ bins, const uint32 bins_w, const uint32 bins_h,
	uint64* const __restrict__ quality_counter, uint64* const __restrict__ pass_counter )
{
	
	// load shared state
	cg::memcpy_async(cg::this_thread_block(), &state, in_state + blockIdx.x, shared_size_bytes);
	cg::wait(cg::this_thread_block());
	
	if(fl::is_block_leader()) {
		state.tss_quality = 0;
		state.tss_passes = 0;
		state.tss_start = clock64();
		state.should_bail = false;
	}
	
	fl::sync_block();
	for( unsigned int i =0; !state.should_bail; i++ ) {
		
		Real opacity = flame_pass(i, shuf_bufs);

		auto transformed = iterator{};
		
		if constexpr(HAS_FINAL_XFORM) {
			iterator my_iter_copy = my_iter();
			dispatch(NUM_XFORMS, my_iter_copy, transformed);
		} else transformed = my_iter();
		
		Real tmp = state.flame[0] * transformed.x + state.flame[2] * transformed.y + state.flame[4];
		transformed.y = trunc(state.flame[1] * transformed.x + state.flame[3] * transformed.y + state.flame[5]);
		transformed.x = trunc(tmp);
		
		transformed.x += state.antialiasing_offsets.x;
		transformed.y += state.antialiasing_offsets.y;
		
		unsigned int hit = 0;
		if(transformed.x >= 0 && transformed.y >= 0 
		&& transformed.x < bins_w && transformed.y < bins_h 
		&& opacity > 0.0) {

			float4& bin = bins[int(transformed.y) * bins_w + int(transformed.x)];
			unsigned char palette_idx = static_cast<unsigned char>( floor(my_iter().z * 255.4999f) );
			float4 new_bin = bin;
			new_bin.x += state.palette[palette_idx].x/255.0f * opacity;
			new_bin.y += state.palette[palette_idx].y/255.0f * opacity;
			new_bin.z += state.palette[palette_idx].z/255.0f * opacity;
			new_bin.w += opacity;
			bin = new_bin;
			hit = (unsigned int)(255.0f * opacity);
		}
		hit = fl::warp_reduce(hit);
		if(fl::is_warp_leader()) {
			atomicAdd(&state.tss_quality, hit);
		}
		
		fl::sync_block();
		if(fl::is_block_leader()) {
			state.should_bail = state.tss_quality > (quality_target / gridDim.x) || (clock64() - state.tss_start) >= time_bailout || i >= iter_bailout;
			state.tss_passes += blockDim.x;
		}
		fl::sync_block();
	}
	
	if(fl::is_block_leader()) {
		atomicAdd(quality_counter, state.tss_quality);
		atomicAdd(pass_counter, state.tss_passes);
	}
	
	// save shared state
	cg::memcpy_async(cg::this_thread_block(), in_state + blockIdx.x, &state, shared_size_bytes);
	cg::wait(cg::this_thread_block());
}
