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
#define my_xform_vote() (state.ts.xform_vote[fl::block_rank()])

constexpr uint32 isqrt_ceil(uint32 n) {
	uint32 a = 1;
	while (a * a < n) a++;
	return a;
}

__device__ __forceinline__ constexpr uint32 feistel_round_fn(uint32 v, uint32 key) {
	v ^= key;
	v ^= v >> 16;
	v *= 0x85ebca6bu;
	v ^= v >> 13;
	v *= 0xc2b2ae35u;
	v ^= v >> 16;
	return v;
}

template<uint32 rounds>
__device__ constexpr uint32 feistel_permute(uint32 x, uint32 seed) {
	constexpr uint32 half = isqrt_ceil(threads_per_block);

	do {
		uint32 L = x / half;
		uint32 R = x % half;

		for (uint32 i = 0; i < rounds; i++) {
			uint32 round_key = seed ^ (i * 0x9e3779b9u);
			uint32 new_R = (L + feistel_round_fn(R, round_key)) % half;
			L = R;
			R = new_R;
		}

		x = L * half + R;
	} while (x >= threads_per_block);

	return x;
}

__device__ void randomize_iterators(const uint32 bins_w, const uint32 bins_h) {
	__shared__ vec2<Real> cp_offset;

	if(fl::is_block_leader()) {
		cp_offset.x = my_rand().rand01() * Real(2.0) - Real(1.0);
		cp_offset.y = my_rand().rand01() * Real(2.0) - Real(1.0);
	}

	// Cranley-Patterson rotation of Hammersley sequence
	auto pos = hammersley::sample<Real, threads_per_block>(fl::block_rank());

	fl::sync_block();
	pos.x += cp_offset.x;
	pos.y += cp_offset.y;
	if(pos.x > Real(1.0)) pos.x -= Real(2.0);
	if(pos.x < Real(-1.0)) pos.x += Real(2.0);
	if(pos.y > Real(1.0)) pos.y -= Real(2.0);
	if(pos.y < Real(-1.0)) pos.y += Real(2.0);

	// Map from [-1,1] to screen space, then to flame space
	pos.x = (pos.x + Real(1.0)) / Real(2.0) * bins_w;
	pos.y = (pos.y + Real(1.0)) / Real(2.0) * bins_h;
	state.flame.plane_space.apply(pos);

	state.ts.iterators[fl::block_rank()] = {pos.x, pos.y, my_rand().rand01()};

	if constexpr(use_chaos) {
		my_xform_vote() = static_cast<unsigned char>(255);
	}

	fl::sync_block();
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
		fl::sync_warp();
	}

	auto& in_local = state.ts.iterators[fl::block_rank()];
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
	
	const auto shuf = feistel_permute<6>(fl::block_rank(), pass_idx);
	fl::sync_block();
	state.ts.iterators[shuf] = out_local;

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

		randomize_iterators(bins_w, bins_h);
		for(unsigned int pass = 0; pass < warmup_count; pass++) {
			auto transformed = flame_pass(pass);

			if constexpr(has_final_xform) {
				vec3<Real> my_iter_copy = {transformed.x, transformed.y, transformed.z};
				state.flame.dispatch(num_xforms, my_iter_copy, transformed, &my_rand());
			}
		
			state.flame.screen_space.apply(transformed.as_vec2());

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

__device__ float4 ld_cg_evict_last(const float4* addr) {
    float4 r;
    asm volatile(
        "{\n\t"
        "  .reg .b64 policy;\n\t"
        "  createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "  ld.global.cg.L2::cache_hint.v4.f32 {%0, %1, %2, %3}, [%4], policy;\n\t"
        "}"
        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
        : "l"(addr)
        : "memory"
    );
    return r;
}

__device__ void st_cg_evict_last(float4* addr, float4 val) {
    asm volatile(
        "{\n\t"
        "  .reg .b64 policy;\n\t"
        "  createpolicy.fractional.L2::evict_last.b64 policy, 1.0;\n\t"
        "  st.global.cg.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, policy;\n\t"
        "}"
        :
        : "l"(addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
        : "memory"
    );
}

__device__ void write_bin(float4* __restrict__ bins, int bin_idx, float4 contribution) {
	if(bin_idx >= 0) {
		float4 bin = ld_cg_evict_last(bins + bin_idx);
		bin.x += contribution.x;
		bin.y += contribution.y;
		bin.z += contribution.z;
		bin.w += contribution.w;
		st_cg_evict_last(bins + bin_idx, bin);
	}
}

__device__ void write_bin_atomic(float4* __restrict__ bins, int bin_idx, float4 contribution) {
	if(bin_idx >= 0) {
		atomicAdd(&bins[bin_idx].x, contribution.x);
		atomicAdd(&bins[bin_idx].y, contribution.y);
		atomicAdd(&bins[bin_idx].z, contribution.z);
		atomicAdd(&bins[bin_idx].w, contribution.w);
	}
}

__device__ void warp_aggregated_write(
    float4* __restrict__ bins,
    int bin_idx,           // -1 if this thread has no valid hit
    float4 contribution    // the RGBA contribution to add
) {
    const uint32 active = __activemask();
    
    const int match_idx = (bin_idx >= 0) ? bin_idx : -1 - (int)threadIdx.x;
    
    const uint32 match_mask = __match_any_sync(active, match_idx);
    const int match_count = __popc(match_mask);
    
    if (__all_sync(active, match_count <= 1)) {
        if (bin_idx >= 0) {
            #ifdef FLAG_ATOMIC
            write_bin_atomic(bins, bin_idx, contribution);
            #else
            write_bin(bins, bin_idx, contribution);
            #endif
        }
        return;
    }
    
    float4 sum = contribution;
    
    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        float4 other;
        other.x = __shfl_xor_sync(match_mask, sum.x, delta);
        other.y = __shfl_xor_sync(match_mask, sum.y, delta);
        other.z = __shfl_xor_sync(match_mask, sum.z, delta);
        other.w = __shfl_xor_sync(match_mask, sum.w, delta);
        
        // Only add if the other thread is in our match group
        uint32 other_lane = (threadIdx.x % 32) ^ delta;
        if (match_mask & (1u << other_lane)) {
            sum.x += other.x;
            sum.y += other.y;
            sum.z += other.z;
            sum.w += other.w;
        }
    }
    
    const int leader = __ffs(match_mask) - 1;
    const bool is_leader = ((threadIdx.x % 32) == leader);
    
    if (bin_idx >= 0 && is_leader) {
        #ifdef FLAG_ATOMIC
        write_bin_atomic(bins, bin_idx, sum);
        #else
        write_bin(bins, bin_idx, sum);
        #endif
    }
}

constexpr static uint32 randomize_interval = 200;
constexpr static uint32 fusion_length = 32;

__device__ unsigned int pass_and_draw(unsigned int pass_idx, float4* const __restrict__ bins, const uint32 bins_w, const uint32 bins_h) {
	
	const auto cycle_pos = (pass_idx + fusion_length) % randomize_interval;

	if(cycle_pos == 0) {
		randomize_iterators(bins_w, bins_h);
	}

	auto transformed = flame_pass(pass_idx);

	if(cycle_pos < fusion_length) {
		return 0;
	}

	if constexpr(has_final_xform) {
			vec3<Real> my_iter_copy = transformed;
			state.flame.dispatch(num_xforms, my_iter_copy, transformed, &my_rand());
	}
		
	state.flame.screen_space.apply(transformed.as_vec2());

	transformed.x = int(roundf(transformed.x));
	transformed.y = int(roundf(transformed.y));

	float4 new_bin = {0.0f, 0.0f, 0.0f, 0.0f};

	auto bin_idx = int(transformed.y) * int(bins_w) + int(transformed.x);
	unsigned int hit = 0;
	if(transformed.x >= 0 && transformed.y >= 0 
	&& transformed.x < bins_w && transformed.y < bins_h 
	&& transformed.w > 0.0) {

		const auto palette_idx = transformed.z * 255.0f;

		const auto& upper = state.palette[static_cast<unsigned char>(ceil(palette_idx))];
		const auto& lower = state.palette[static_cast<unsigned char>(floor(palette_idx))];
		auto mix = palette_idx - truncf(palette_idx);
		auto factor = transformed.w / 255.0f;

		new_bin.x += ((1.0_r - mix) * lower.x + mix * upper.x) * factor;
		new_bin.y += ((1.0_r - mix) * lower.y + mix * upper.y) * factor;
		new_bin.z += ((1.0_r - mix) * lower.z + mix * upper.z) * factor;
		new_bin.w += transformed.w;

		hit = (unsigned int)(255.0f * transformed.w);
	} else {
		bin_idx = -1;
	}

	#ifdef FLAG_WARP_AGGREGATED_WRITE
	warp_aggregated_write(bins, bin_idx, new_bin);
	#else
	#ifdef FLAG_ATOMIC
	write_bin_atomic(bins, bin_idx, new_bin);
	#else
	write_bin(bins, bin_idx, new_bin);
	#endif
	#endif

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
	unsigned int* const __restrict__ sample_indices,
	unsigned long long* const __restrict__ warp_collisions)
{
	
	decltype(clock64()) start_time;
	if(fl::is_block_leader()) {
		start_time = clock64();
		iter_info.init(temporal_multiplier, temporal_slicing);
		atomicMin(earliest_start, fl::time());

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
			iter_info.bail |= iter_info.iter >= iter_bailout ||
				//iter_info.iter >= temporal_slicing
				// ((iter_info.on_sample_boundary() && iter_info.lowest_active_sample() == iter_info.current_sample) || temporal_multiplier == 1)
				 ((fl::time() - iter_info.start_time) >= time_bailout || *stop_render);
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
		atomicMax(warp_collisions, clock64() - start_time);
		atomicMax(latest_stop, fl::time());
	}
	
}
