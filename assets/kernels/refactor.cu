#define USE_ASYNC_MEMCPY

#ifdef USE_ASYNC_MEMCPY
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;
#endif

#define DEBUG_GRID(...) if(threadIdx.x == 0 && blockIdx.x == 0) {printf(__VA_ARGS__);}
#define DEBUG_BLOCK(format, ...) if(threadIdx.x == 0) {printf("block %d: " format "\n", blockIdx.x, __VA_ARGS__);}

#include <refrakt/flamelib.h>
#include <refrakt/random.h>

namespace hammersley {
	
	namespace detail {
		
		constexpr __device__ unsigned int ReverseBits32(unsigned int n) {
			n = (n << 16) | (n >> 16);
			n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
			n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
			n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
			n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
			return n;
		}
		
		constexpr __device__ unsigned int flip_bits(unsigned int v, unsigned int radix) {
			return ReverseBits32(v) >> (32 - radix);
		}

		constexpr __device__ unsigned int next_pow2(unsigned int v) {
			v--;
			v |= v >> 1;
			v |= v >> 2;
			v |= v >> 4;
			v |= v >> 8;
			v |= v >> 16;
			v++;
			return v;
		}
	}
	
	template<typename Real, unsigned int Total>
	__device__ constexpr vec2<Real> sample( unsigned int idx) {
		Real inv_max;
		
		unsigned radix = 0;
		unsigned int max_val = (Total % 2 == 0) ? Total: detail::next_pow2(Total);
		unsigned int v = max_val;
		while(v >>= 1) radix++;
		
		inv_max = static_cast<Real>(1.0)/max_val;
		
		return {
			idx * inv_max * static_cast<Real>(2.0) - static_cast<Real>(1.0),
			detail::flip_bits(idx, radix) * inv_max * static_cast<Real>(2.0) - static_cast<Real>(1.0)
		};
	}
}

namespace fl = flamelib;

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
	print_sizes<float, 768, 512, 384, 256, 192, 128, 96>();
}

__device__ unsigned short shuf_bufs[THREADS_PER_BLOCK * NUM_SHUF_BUFS];

__device__ 
void queue_shuffle_load(uint32 pass_idx) {
	if(pass_idx % threads_per_block == 0) {
		my_shuffle_vote() = my_rand().rand() % num_shuf_bufs;
	}
	
	fl::sync_block();
	#ifdef USE_ASYNC_MEMCPY
	cg::memcpy_async(cg::this_thread_block(), 
		state.ts.shuffle, 
		shuf_bufs + threads_per_block * state.ts.shuffle_vote[pass_idx % threads_per_block], 
		shuf_buf_size);
	#endif
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
			constexpr static auto hsv_to_rgb = 
			[]( const Real h, const Real s, const Real v ) -> uchar3 {
				auto f = [&](Real n) {
					auto k = fmodf(n + h / Real(60.0), Real(6.0));
					return v - v * s * max(Real(0.0), min(k, min(Real(4.0) - k, Real(1.0))));
				};

				return {
					static_cast<unsigned char>(f(Real(5.0)) * Real(255.0)),
					static_cast<unsigned char>(f(Real(3.0)) * Real(255.0)),
					static_cast<unsigned char>(f(Real(1.0)) * Real(255.0)),
				};
			};

			palette[idx] = 
				hsv_to_rgb(
					seg[3 * idx].sample(t),
					seg[3 * idx + 1].sample(t),
					seg[3 * idx + 2].sample(t)
				);
		}
	}
	
}

template<uint32 count>
void memcpy_sync(const uint8* const __restrict__ src, uint8* const __restrict__ dest) {
	constexpr static auto vector_count = count / sizeof(int4);
	constexpr static auto leftover = count % sizeof(int4);

	constexpr static uint32 per_thread = vector_count / threads_per_block + ((vector_count % threads_per_block)? 1: 0);

	for(uint32 i = 0; i < per_thread; i++) {
		const auto idx = i * threads_per_block + fl::block_rank();
		if(idx < vector_count) {
			((int4*)dest)[idx] = ((int4*)src)[idx];
		}
	}

    if(flamelib::is_block_leader()) {
		for(uint32 i = count - leftover; i < count; i++) {
			dest[i] = src[i];
		}
	}
}

__device__ 
vec4<Real> flame_pass(unsigned int pass_idx) {
	
	// every 32 passes, repopulate this warp's xid buffer
	if(pass_idx % 32 == 0) {
		my_xform_vote() = state.flame.select_xform(my_rand().rand01());
	}
	fl::sync_warp();

	auto in_local = iterator{my_iter(x), my_iter(y), my_iter(color)};
	auto out_local = iterator{-666.0, -666.0, -660.0};
	auto selected_xform = state.ts.xform_vote[fl::warp_start_in_block() + pass_idx % 32];

	Real opacity = state.flame.dispatch( 
		selected_xform, 
		in_local, out_local, &my_rand()
	);

	if(badvalue(out_local.x) || badvalue(out_local.y)) {
		out_local.x = my_rand().rand01() * 2.0 - 1.0;
		out_local.y = my_rand().rand01() * 2.0 - 1.0;
		opacity = 0.0;
	}
	
	#ifdef USE_ASYNC_MEMCPY
	cg::wait(cg::this_thread_block());
	state.ts.iterators.x[my_shuffle()] = out_local.x;
	state.ts.iterators.y[my_shuffle()] = out_local.y;
	state.ts.iterators.color[my_shuffle()] = out_local.z;
	#else
	fl::sync_block();
	const auto shuf = shuf_bufs[threads_per_block * state.ts.shuffle_vote[pass_idx % threads_per_block] + fl::block_rank()];
	state.ts.iterators.x[shuf] = out_local.x;
	state.ts.iterators.y[shuf] = out_local.y;
	state.ts.iterators.color[shuf] = out_local.z;
	#endif

	queue_shuffle_load(pass_idx + 1);

	return vec4<Real>{out_local.x, out_local.y, out_local.z, opacity};
}

__global__ 
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_MP)
void warmup(
	const uint32 num_segments,
	const segment* const __restrict__ segments, 
	const uint32 seed, const uint32 warmup_count, const uint32 bins_w, const uint32 bins_h,
	shared_state_t* __restrict__ out_state ) 
{	
	my_rand().init(seed + fl::grid_rank());
	
	queue_shuffle_load(0);
	
	const auto seg_size = gridDim.x / num_segments;
	const auto t = (blockIdx.x % seg_size)/Real(gridDim.x/ num_segments);
	const auto seg = blockIdx.x / seg_size;
	const auto seg_offset = flame_size_reals + palette_size;
	
	//DEBUG_BLOCK("%f, %d, %d, %d", t, blockIdx.x, gridDim.x, num_segments);

	interpolate_flame<flame_size_reals>(t, segments + (seg_offset) * seg, state.flame.as_array());
	interpolate_palette<256>(t, segments + flame_size_reals + (seg_offset) * seg, state.palette);
	
	if(fl::is_block_leader()) {
		state.flame.do_precalc(&my_rand());
	}

	auto pos = hammersley::sample<Real, TOTAL_THREADS>(fl::grid_rank());
	pos.x = (pos.x + 1)/2 * bins_w;
	pos.y = (pos.y + 1)/2 * bins_h;
	state.flame.plane_space.apply(pos.x, pos.y);
	
	my_iter(x) = pos.x;
	my_iter(y) = pos.y;
	my_iter(color) = my_rand().rand01();

	fl::sync_block();
	for(unsigned int pass = 0; pass < warmup_count; pass++) {
		flame_pass(pass);
	}

	if(fl::is_grid_leader()) {
		//state.flame.print_debug();
	}

	// save shared state
	#ifdef USE_ASYNC_MEMCPY
	cg::memcpy_async(cg::this_thread_block(), out_state + blockIdx.x, &state, shared_size_bytes);
	cg::wait(cg::this_thread_block());
	#else
	memcpy_sync<shared_size_bytes>((uint8*)&state, (uint8*)(out_state + blockIdx.x));
	#endif
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
	uint64* const __restrict__ quality_counter, uint64* const __restrict__ pass_counter)
{
	
	// load shared state
	#ifdef USE_ASYNC_MEMCPY
	cg::memcpy_async(cg::this_thread_block(), &state, in_state + blockIdx.x, shared_size_bytes);
	cg::wait(cg::this_thread_block());
	#else
	memcpy_sync<shared_size_bytes>((uint8*)(in_state + blockIdx.x), (uint8*)&state);
	#endif

	if(fl::is_block_leader()) {
		state.tss_quality = 0;
		state.tss_passes = 0;
		state.tss_start = fl::time();
		state.should_bail = false;
	}
	
	fl::sync_block();
	for( unsigned int i =0; !state.should_bail; i++ ) {
		
		auto transformed = flame_pass(i);
		
		if constexpr(has_final_xform) {
			vec3<Real> my_iter_copy = {transformed.x, transformed.y, transformed.z};
			state.flame.dispatch(num_xforms, my_iter_copy, transformed, &my_rand());
		}
		
		state.flame.screen_space.apply(transformed.x, transformed.y);

		transformed.x = trunc(transformed.x);
		transformed.y = trunc(transformed.y);
	
		unsigned int hit = 0;
		if(transformed.x >= 0 && transformed.y >= 0 
		&& transformed.x < bins_w && transformed.y < bins_h 
		&& transformed.w > 0.0) {

			float4& bin = bins[int(transformed.y) * bins_w + int(transformed.x)];

			const auto palette_idx = transformed.z * 255.0f;

			const auto& upper = state.palette[static_cast<unsigned char>(ceil(palette_idx))];
			const auto& lower = state.palette[static_cast<unsigned char>(floor(palette_idx))];
			auto mix = palette_idx - truncf(palette_idx);

			float4 new_bin = bin;

			new_bin.x += ((1.0_r - mix) * lower.x + mix * upper.x) / 255.0f * transformed.w;
			new_bin.y += ((1.0_r - mix) * lower.y + mix * upper.y) / 255.0f * transformed.w;
			new_bin.z += ((1.0_r - mix) * lower.z + mix * upper.z) / 255.0f * transformed.w;
			new_bin.w += transformed.w;
			bin = new_bin;
			hit = (unsigned int)(255.0f * transformed.w);

		}
		hit = fl::warp_reduce(hit);
		if(fl::is_warp_leader()) {
			atomicAdd(&state.tss_quality, hit);
		}
		
		fl::sync_block();
		if(fl::is_block_leader()) {
			state.should_bail = state.tss_quality > (double(quality_target) / gridDim.x) || (fl::time() - state.tss_start) >= time_bailout || i >= iter_bailout;
			state.tss_passes += blockDim.x;
		}
		fl::sync_block();
	}
	
	if(fl::is_block_leader()) {
		atomicAdd(quality_counter, state.tss_quality);
		atomicAdd(pass_counter, state.tss_passes);
		//DEBUG_BLOCK("quality: %u, passes: %u", state.tss_quality, state.tss_passes);
	}
	
	// save shared state
	#ifdef USE_ASYNC_MEMCPY
	cg::memcpy_async(cg::this_thread_block(), in_state + blockIdx.x, &state, shared_size_bytes);
	cg::wait(cg::this_thread_block());
	#else
	memcpy_sync<shared_size_bytes>((uint8*)&state, (uint8*)(in_state + blockIdx.x));
	#endif
}
