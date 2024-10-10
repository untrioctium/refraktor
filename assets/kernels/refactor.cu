#include <refrakt/flamelib.h>
#include <refrakt/random.h>

#ifdef USE_CHAOS
constexpr static bool use_chaos = true;
#else
constexpr static bool use_chaos = false;
#endif

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

struct exec_config {
	uint64 grid;
	uint64 block;
	uint64 shared_per_block;
};

#include "flame_generated.h"

using iterator = vec3<Real>;

using sample_state_t = fl::sample_state_tmpl<flame_t<Real, xoroshiro64<Real>>, Real, xoroshiro64<Real>, threads_per_block>;
constexpr auto sample_state_size_bytes = sizeof(sample_state_t);
static_assert(sizeof(sample_state_t::flame) == flame_size_bytes);

__global__ void get_sample_state_size(uint64* out) {
	*out = sample_state_size_bytes;
}

struct segment {
	double a, b, c, d;
	
	__device__ double sample(double t) const { return a * t * t * t + b * t * t + c * t + d; }
};

__shared__ sample_state_t state;
__shared__ fl::iteration_info_t iter_info;

#define my_iter(comp) (state.ts.iterators.comp[fl::block_rank()])
#define my_rand() (state.ts.rand_states[fl::block_rank()])
#define my_shuffle() (state.ts.shuffle[fl::block_rank()])
#define my_shuffle_vote() (state.ts.shuffle_vote[fl::block_rank()])
#define my_xform_vote() (state.ts.xform_vote[fl::block_rank()])

__device__ unsigned short shuf_bufs[THREADS_PER_BLOCK * NUM_SHUF_BUFS];

__device__ void queue_shuffle_load(uint32 pass_idx) {
	if(pass_idx % threads_per_block == 0) {
		my_shuffle_vote() = my_rand().rand() % num_shuf_bufs;
	}
}

template<uint32 count>
__device__ void interpolate_flame( Real t, const segment* const __restrict__ seg, Real* const __restrict__ out ) {
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
__device__ void interpolate_palette( Real t, const segment* const __restrict__ seg, uchar3* const __restrict__ palette )
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
__device__ void memcpy_sync(const uint8* const __restrict__ src, uint8* const __restrict__ dest) {
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

	fl::sync_block();
}

#ifndef USE_CHAOS
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
	
	fl::sync_block();
	const auto shuf = shuf_bufs[threads_per_block * state.ts.shuffle_vote[pass_idx % threads_per_block] + fl::block_rank()];
	state.ts.iterators.x[shuf] = out_local.x;
	state.ts.iterators.y[shuf] = out_local.y;
	state.ts.iterators.color[shuf] = out_local.z;

	queue_shuffle_load(pass_idx + 1);

	return vec4<Real>{out_local.x, out_local.y, out_local.z, opacity};
}
#else
__device__
vec4<Real> flame_pass(unsigned int pass_idx) {

	auto in_local = iterator{my_iter(x), my_iter(y), my_iter(color)};
	auto out_local = iterator{-666.0, -666.0, -660.0};
	auto selected_xform = state.flame.select_xform(my_xform_vote(), my_rand().rand01()); 

	my_xform_vote() = selected_xform;

	Real opacity = state.flame.dispatch( 
		selected_xform, 
		in_local, out_local, &my_rand()
	);

	if(badvalue(out_local.x) || badvalue(out_local.y)) {
		out_local.x = my_rand().rand01() * 2.0 - 1.0;
		out_local.y = my_rand().rand01() * 2.0 - 1.0;
		opacity = 0.0;
	}

	my_iter(x) = out_local.x;
	my_iter(y) = out_local.y;
	my_iter(color) = out_local.z;

	fl::sync_block();
	return vec4<Real>{out_local.x, out_local.y, out_local.z, opacity};
}
#endif

#ifdef ROCCU_CUDA
#define LAUNCH_BOUNDS(TPB, BPMP) __launch_bounds__(TPB, BPMP)
#else
#define LAUNCH_BOUNDS(TPB, BPMP) __launch_bounds__(TPB, (BPMP * TPB) / warpSize)
#endif

__global__ 
LAUNCH_BOUNDS(THREADS_PER_BLOCK, BLOCKS_PER_MP)
void warmup(
	const uint32 num_segments,
	const segment* const __restrict__ segments, 
	const uint32 seed, const uint32 warmup_count, const uint32 bins_w, const uint32 bins_h,
	sample_state_t* __restrict__ out_state,
	const int32 temporal_multiplier, unsigned long long* warmup_hits)  
{	
	auto nsamples = temporal_multiplier * gridDim.x;
	for(int sample = 0; sample < temporal_multiplier; sample++) {
		my_rand().init(seed + fl::grid_rank());
		
		if constexpr(!use_chaos) {
			queue_shuffle_load(0);
		} else {
			my_xform_vote() = static_cast<unsigned char>(255);
		}

		const auto sample_idx = sample * gridDim.x + blockIdx.x;
		const auto seg_size = nsamples / num_segments;
		const auto t = (sample_idx % seg_size)/Real(nsamples/ num_segments);
		//DEBUG_BLOCK("%d %f", sample_idx, t);
		const auto seg = sample_idx / seg_size;
		const auto seg_offset = flame_size_reals + palette_size;
		
		//DEBUG_BLOCK("%f, %d, %d, %d", t, blockIdx.x, gridDim.x, num_segments);

		interpolate_flame<flame_size_reals>(t, segments + (seg_offset) * seg, state.flame.as_array());
		interpolate_palette<256>(t, segments + flame_size_reals + (seg_offset) * seg, state.palette);

		// convert affines from polar while we are still in double land
		if(fl::block_rank() < num_affines) {

			Real* flame_array = state.flame.as_array();

			const auto aff = affine_indices[fl::block_rank()];

			double angx = segments[aff].sample(t);
			double angy = segments[aff + 1].sample(t);
			double magx = exp(segments[aff + 2].sample(t));
			double magy = exp(segments[aff + 3].sample(t));

			flame_array[aff] = magx * cos(angx);
			flame_array[aff + 1] = magx * sin(angx);
			flame_array[aff + 2] = magy * cos(angy);
			flame_array[aff + 3] = magy * sin(angy);
		}

		if(fl::is_block_leader()) {
			state.warmup_hits = 0;
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
			auto transformed = flame_pass(pass);

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
				&& transformed.w > 0.0) 
			{
				hit = (unsigned int)(255.0f * transformed.w);
			}
			hit = fl::warp_reduce(hit);
			if(fl::is_warp_leader()) {
				atomicAdd(&state.warmup_hits, hit);
			}
		}

		if(fl::is_block_leader()) {
			atomicAdd(warmup_hits, state.warmup_hits);
		}

		memcpy_sync<sample_state_size_bytes>((uint8*)&state, (uint8*)(out_state + sample_idx));
	}
}

constexpr static uint64 per_block = THREADS_PER_BLOCK;

__device__ unsigned int pass_and_draw(unsigned int pass_idx, float4* const __restrict__ bins, const uint32 bins_w, const uint32 bins_h) {
	
	auto transformed = flame_pass(pass_idx);
	
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

	return hit;
}

__global__
LAUNCH_BOUNDS(per_block, BLOCKS_PER_MP)
void bin(
	sample_state_t* const __restrict__ in_state,
	const uint64 quality_target,
	const uint32 iter_bailout,
	const uint64 time_bailout,
	float4* const __restrict__ bins, const uint32 bins_w, const uint32 bins_h,
	uint64* const __restrict__ quality_counter, uint64* const __restrict__ pass_counter,
	volatile bool* __restrict__ stop_render,
	const int32 temporal_multiplier,
	const int32 temporal_slicing,
	const unsigned long long* const __restrict__ warmup_hits,
	unsigned long long* const __restrict__ earliest_start,
	unsigned long long* const __restrict__ latest_stop,
	unsigned int* const __restrict__ sample_indices)
{
	
	if(fl::is_block_leader()) {
		iter_info.init(temporal_multiplier, temporal_slicing);
		atomicMin(earliest_start, iter_info.start_time);

		for(int i = 0; i < temporal_multiplier; i++) {
			iter_info.sample_indices[i] = sample_indices[blockIdx.x + gridDim.x * i];
		}
	}

	if(fl::block_rank() < temporal_multiplier) {
		auto& sample = in_state[iter_info.sample_indices[fl::block_rank()]];
		sample.tss_quality = 0;
		sample.tss_passes = 0;
	}
	fl::sync_block();
	
	while(!iter_info.bail) {
		
		if(iter_info.on_sample_boundary() && iter_info.current_sample != iter_info.loaded_sample) {

			if(iter_info.loaded_sample != 0xFFFFFFFF) {
				memcpy_sync<sample_state_size_bytes>((uint8*)&state, (uint8*)(in_state + iter_info.sample_indices[iter_info.loaded_sample]));
				//DEBUG_GRID("saved sample %d\n", iter_info.loaded_sample);
			}

			memcpy_sync<sample_state_size_bytes>((uint8*)(in_state + iter_info.sample_indices[iter_info.current_sample]), (uint8*)&state);
			//DEBUG_GRID("loaded sample %d\n", iter_info.current_sample);
			iter_info.loaded_sample = iter_info.current_sample;
		}

		unsigned int hit = pass_and_draw(iter_info.iter, bins, bins_w, bins_h);
		hit = fl::warp_reduce(hit);
		if(fl::is_warp_leader()) {
			atomicAdd(&state.tss_quality, hit);
		}
		
		fl::sync_block();
		if(fl::is_block_leader()) {

			if(state.tss_quality >= double(quality_target) / (temporal_multiplier * gridDim.x)) {
				iter_info.mark_sample_done();
			}

			state.tss_passes += blockDim.x;

			iter_info.tick();

			iter_info.bail |= iter_info.samples_active == 0;
			iter_info.bail |= 
				iter_info.iter >= temporal_slicing
				&& ((iter_info.on_sample_boundary() && iter_info.lowest_active_sample() == iter_info.current_sample) || temporal_multiplier == 1)
				&& ((fl::time() - iter_info.start_time) >= time_bailout || *stop_render);
		}
		fl::sync_block();
	}

	memcpy_sync<sample_state_size_bytes>((uint8*)&state, (uint8*)(in_state + iter_info.sample_indices[iter_info.loaded_sample]));

	
	if(fl::block_rank() < temporal_multiplier) {
		auto& sample = in_state[iter_info.sample_indices[fl::block_rank()]];
		atomicAdd(quality_counter, sample.tss_quality);
		atomicAdd(pass_counter, sample.tss_passes);
	}
	
	if(fl::is_block_leader()) {
		atomicMax(latest_stop, fl::time());
	}
	
}
