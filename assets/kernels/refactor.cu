#include <cooperative_groups/memcpy_async.h>

#define DEBUG_GRID(...) if(threadIdx.x == 0 && blockIdx.x == 0) {printf(__VA_ARGS__);}
#define DEBUG_BLOCK(format, ...) if(threadIdx.x == 0) {printf("block %d: " format "\n", blockIdx.x, __VA_ARGS__);}

#include <refrakt/precision.h>
#include <refrakt/random.h>
#include <refrakt/flamelib.h>
#include <refrakt/hammersley.h>
#include <refrakt/color.h>

namespace fl = flamelib;
namespace cg = cooperative_groups;

constexpr static uint32 threads_per_block = THREADS_PER_BLOCK;
constexpr static uint32 flame_size_reals = FLAME_SIZE_REALS;
constexpr static uint32 flame_size_bytes = flame_size_reals * sizeof(Real);
constexpr static uint32 num_shuf_bufs = NUM_SHUF_BUFS;
constexpr static uint32 shuf_buf_size = threads_per_block * sizeof(uint16);

struct exec_config {
	uint64 grid;
	uint64 block;
	uint64 shared_per_block;
};

#include "flame_generated.h"

using iterator = vec3<Real>;

using shared_state_t = fl::shared_state_tmpl<flame_t<Real, xoroshiro64<Real>>, Real, xoroshiro64<Real>, threads_per_block>;
constexpr auto shared_size_bytes = sizeof(shared_state_t);
static_assert(sizeof(shared_state_t::flame) == flame_size_bytes);

struct segment {
	double a, b, c, d;
	
	Real sample(Real t) const { return Real(a * t * t * t + b * t * t + c * t + d); }
};

__shared__ shared_state_t state;

#define my_iter(comp) (state.ts.iterators. ## comp ## [fl::block_rank()])
#define my_rand() (state.ts.rand_states[fl::block_rank()])
#define my_shuffle() (state.ts.shuffle[fl::block_rank()])
#define my_shuffle_vote() (state.ts.shuffle_vote[fl::block_rank()])
#define my_xform_vote() (state.ts.xform_vote[fl::block_rank()])

template<typename Real, uint64... ThreadsPerBlock>
void print_sizes() {

	//(printf("%llu: %llu\n", ThreadsPerBlock, sizeof(cub::BlockRadixSort<uint64, ThreadsPerBlock, 4>::TempStorage)), ...);

}

__global__
void print_debug_info() {
	/*#define output_sizeof(T) printf("sizeof(" #T ")=%llu (%.2f%% of total shared)\n", sizeof(T), sizeof(T) / float(sizeof(shared_state_t)) * 100.0f);
	
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

	output_sizeof(flame_t<float>);

	printf("bytes per iterator: %llu (%llu leftover)\n", sizeof(thread_states_t)/THREADS_PER_BLOCK, sizeof(thread_states_t) % THREADS_PER_BLOCK);
	printf("shared state size - flame size - iterators size = %llu\n", sizeof(shared_state_t) - sizeof(shared_state_t::flame) - sizeof(thread_states_t));*/

	print_sizes<float, 768, 512, 384, 256, 192, 128, 96>();
}

__device__ unsigned short shuf_bufs[THREADS_PER_BLOCK * NUM_SHUF_BUFS];

__device__ 
void queue_shuffle_load(uint32 pass_idx) {
	if(pass_idx % threads_per_block == 0) {
		my_shuffle_vote() = my_rand().rand() % num_shuf_bufs;
	}
	
	fl::sync_block();
	cg::memcpy_async(cg::this_thread_block(), 
		state.ts.shuffle, 
		shuf_bufs + threads_per_block * state.ts.shuffle_vote[pass_idx % threads_per_block], 
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
Real flame_pass(unsigned int pass_idx, uint64* const xform_counters = nullptr) {
	
	// every 32 passes, repopulate this warp's xid buffer
	if(pass_idx % 32 == 0) {
		my_xform_vote() = state.flame.select_xform(my_rand().rand_uniform());
	}
	fl::sync_warp();

	auto out_local = iterator{-666.0, -666.0, -660.0};
	//DEBUG("xid: %d\n", ts[warp.meta_group_rank() * 32 + pass_idx % 32].xid);
	auto selected_xform = state.ts.xform_vote[fl::warp_start_in_block() + pass_idx % 32];
	//if(fl::is_warp_leader() && xform_counters != nullptr) atomicAdd(xform_counters + selected_xform, fl::warp_size());
	Real opacity = state.flame.dispatch( 
		selected_xform, 
		my_iter(x), my_iter(y), my_iter(color), out_local.x, out_local.y, out_local.z, &my_rand()
	);
	//out_local = iterator{{-666.0, -666.0}, -660.0};

	//DEBUG("iter(%f,%f,%f)\n", my_iter(x), my_iter(y), my_iter(color));

	if(badvalue(out_local.x) || badvalue(out_local.y)) {
		out_local.x = my_rand().rand_uniform() * 2.0 - 1.0;
		out_local.y = my_rand().rand_uniform() * 2.0 - 1.0;
		opacity = 0.0;
	}
	
	cg::wait(cg::this_thread_block());
	state.ts.iterators.x[my_shuffle()] = out_local.x;
	state.ts.iterators.y[my_shuffle()] = out_local.y;
	state.ts.iterators.color[my_shuffle()] = out_local.z;
	queue_shuffle_load(pass_idx + 1); // note: implicit block.sync()
	return opacity;
}


__global__ 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
void warmup(
	const uint32 num_segments,
	const segment* const __restrict__ segments, 
	const uint32 seed, const uint32 warmup_count,
	shared_state_t* __restrict__ out_state ) 
{	
	my_rand().init(seed + fl::grid_rank());
	
	my_iter(x) = my_rand().rand_uniform();
	my_iter(y) = my_rand().rand_uniform();
	my_iter(color) = my_rand().rand_uniform();
	
	if(fl::is_block_leader()) {
		state.antialiasing_offsets = my_rand().rand_gaussian(1/2.0);
	}
	queue_shuffle_load(0);
	
	const auto seg_size = gridDim.x / num_segments;
	const auto t = (blockIdx.x % seg_size)/Real(gridDim.x/ num_segments);
	const auto seg = blockIdx.x / seg_size;
	const auto seg_offset = flame_size_reals + palette_size;
	
	//DEBUG_BLOCK("%f, %d, %d, %d", t, blockIdx.x, gridDim.x, num_segments);

	interpolate_flame<flame_size_reals>(t, segments + (seg_offset) * seg, state.flame.as_array());
	interpolate_palette<256>(t, segments + flame_size_reals + (seg_offset) * seg, state.palette);
	
	if(fl::is_block_leader()) state.flame.do_precalc();

	fl::sync_block();
	for(unsigned int pass = 0; pass < warmup_count; pass++) {
		flame_pass(pass);
	}

	// save shared state
	cg::memcpy_async(cg::this_thread_block(), out_state + blockIdx.x, &state, shared_size_bytes);
	cg::wait(cg::this_thread_block());
}

constexpr static uint64 per_block = THREADS_PER_BLOCK;

__global__
__launch_bounds__(per_block, BLOCKS_PER_MP)
void bin(
	shared_state_t* const __restrict__ in_state,
	const uint64 quality_target,
	const uint32 iter_bailout,
	const uint64 time_bailout,
	float4* const __restrict__ bins, const uint32 bins_w, const uint32 bins_h,
	uint64* const __restrict__ quality_counter, uint64* const __restrict__ pass_counter,
	uint64* const __restrict__ xform_counters )
{
	
	// load shared state
	cg::memcpy_async(cg::this_thread_block(), &state, in_state + blockIdx.x, shared_size_bytes);
	cg::wait(cg::this_thread_block());
	
	if(fl::is_block_leader()) {
		state.tss_quality = 0;
		state.tss_passes = 0;
		state.tss_start = fl::time();
		state.should_bail = false;
	}
	
	fl::sync_block();
	for( unsigned int i =0; !state.should_bail; i++ ) {
		
		Real opacity = flame_pass(i, xform_counters);

		auto transformed = vec3<Real>{};
		
		if constexpr(has_final_xform) {
			vec3 my_iter_copy = {my_iter(x), my_iter(y), my_iter(color)};
			state.flame.dispatch(num_xforms, my_iter_copy.x, my_iter_copy.y, my_iter_copy.z, transformed.x, transformed.y, transformed.z, &my_rand());
		} else transformed = {my_iter(x), my_iter(y), my_iter(color)};
		
		state.flame.screen_space.apply(transformed.x, transformed.y);
		
		transformed.x += state.antialiasing_offsets.x;
		transformed.y += state.antialiasing_offsets.y;

		transformed.x = trunc(transformed.x);
		transformed.y = trunc(transformed.y);
		
		unsigned int hit = 0;
		if(transformed.x >= 0 && transformed.y >= 0 
		&& transformed.x < bins_w && transformed.y < bins_h 
		&& opacity > 0.0) {

			const auto palette_idx = transformed.z * 255.0f;
			auto bin_idx = int(transformed.y) * bins_w + int(transformed.x);
			const auto& upper = state.palette[static_cast<unsigned char>(ceil(palette_idx))];
			const auto& lower = state.palette[static_cast<unsigned char>(floor(palette_idx))];
			auto mix = palette_idx - truncf(palette_idx);

			ushort4 result{
				static_cast<unsigned short>(((1.0_r - mix) * lower.x + mix * upper.x) * opacity),
				static_cast<unsigned short>(((1.0_r - mix) * lower.y + mix * upper.y) * opacity),
				static_cast<unsigned short>(((1.0_r - mix) * lower.z + mix * upper.z) * opacity),
				static_cast<unsigned short>(255.0f * opacity)
			};

//			DEBUG_BLOCK("%d %d %d %d", result.x, result.y, result.z, result.w);
			float4& bin = bins[bin_idx];

			float4 new_bin = bin;
			new_bin.x += result.x/255.0f;
			new_bin.y += result.y/255.0f;
			new_bin.z += result.z/255.0f;
			new_bin.w += result.w/255.0f;
			bin = new_bin;
			
			hit = (unsigned int)(result.w);
		}
		hit = fl::warp_reduce(hit);
		if(fl::is_warp_leader()) {
			atomicAdd(&state.tss_quality, hit);
		}
		
		fl::sync_block();
		if(fl::is_block_leader()) {
			state.should_bail = state.tss_quality > (quality_target / gridDim.x) || (fl::time() - state.tss_start) >= time_bailout || i >= iter_bailout;
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
