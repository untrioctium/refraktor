#pragma once

#include <typeinfo>
#include <future>

#include <ezrtc.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_types.h>
#include <librefrakt/gpu_buffer.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/hash.h>

#include <librefrakt/allocators.h>

#include <librefrakt/constants.h>

namespace rfkt {

	class flame_compiler;

	class flame_kernel: public traits::noncopyable {

	public:
		friend class flame_compiler;

		struct bin_result {
			double quality;
			double elapsed_ms;
			std::size_t total_passes;
			std::size_t total_draws;
			std::size_t total_bins;
			double passes_per_thread;
			double max_density;
		};

		struct saved_state: public traits::noncopyable {
			roccu::gpu_image<float4> bins = {};
			double quality = 0.0;
			int temporal_multiplier = 1;
			roccu::gpu_buffer<> shared = {};
			roccu::gpu_span<bool> stopper;
			roccu::gpu_buffer<std::size_t> warmup_hits = {};
			std::future<double> warmup_time;
			roccu::gpu_buffer<std::size_t> density_histogram;

			saved_state() = default;
			saved_state(saved_state&& o) noexcept {
				(*this) = std::move(o);
			}
			saved_state& operator=(saved_state&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(shared, o.shared);
				std::swap(quality, o.quality);
				std::swap(stopper, o.stopper);
				std::swap(temporal_multiplier, o.temporal_multiplier);
				std::swap(warmup_hits, o.warmup_hits);
				std::swap(warmup_time, o.warmup_time);
				std::swap(density_histogram, o.density_histogram);
				return *this;
			}

			saved_state(uint2 dims, std::size_t nbytes, int temporal_multiplier, std::future<double>&& warmup_time, RUstream stream) :
				bins(dims, stream),
				temporal_multiplier(temporal_multiplier),
				shared(nbytes * temporal_multiplier, stream),
				warmup_hits(1, stream),
				warmup_time(std::move(warmup_time)),
				density_histogram(histogram_size, stream){
				bins.clear(stream);
				warmup_hits.clear(stream);
			}

			saved_state(decltype(saved_state::bins)&& bins, std::size_t nbytes, int temporal_multiplier, std::future<double>&& warmup_time, RUstream stream) :
				bins(std::move(bins)),
				temporal_multiplier(temporal_multiplier),
				shared(nbytes * temporal_multiplier, stream),
				warmup_hits(1, stream),
				warmup_time(std::move(warmup_time)),
				density_histogram(histogram_size, stream){
				bins.clear(stream);
				warmup_hits.clear(stream);
			}

			void abort_binning() {
				const static bool stop = true;
				stopper.from_host(std::span<const bool>{ &stop, 1 }, nullptr);
			}

		};

		struct bailout_args {
			std::uint32_t iters = 4'000'000'000;
			std::uint32_t millis = 1000;
			double quality = 128.0;
		};

		auto bin(roccu::gpu_stream& stream, flame_kernel::saved_state& state, const bailout_args&, int temporal_slicing = 100) const-> std::future<bin_result>;
		auto warmup(roccu::gpu_stream& stream, std::span<double> samples, uint2 dims, std::uint32_t seed, std::uint32_t count, int temporal_multiplier = 1) const->flame_kernel::saved_state;
		auto warmup(roccu::gpu_stream& stream, std::span<double> samples, roccu::gpu_image<float4>&& bins, std::uint32_t seed, std::uint32_t count, int temporal_multiplier = 1) const->flame_kernel::saved_state;

		flame_kernel(flame_kernel&& o) noexcept {
			*this = std::move(o);
		}

		flame_kernel& operator=(flame_kernel&& o) noexcept {
			this->exec = o.exec;
			this->mod = std::move(o.mod);
			this->flame_size_reals = o.flame_size_reals;
			this->srt = std::move(o.srt);
			this->affine_indices = std::move(o.affine_indices);
			this->saved_state_size = o.saved_state_size;

			return *this;
		}

		flame_kernel() = default;

		bool valid() const {
			return mod.operator bool();
		}

	private:

		struct shared_runtime {
			ezrtc::cuda_module catmull = {};
			ezrtc::cuda_module histogram = {};
			rfkt::pinned_ring_allocator pra;
			rfkt::device_ring_allocator dra;
		};

		flame_kernel(std::size_t flame_size_reals, ezrtc::cuda_module&& mod, std::pair<int, int> exec, std::shared_ptr<shared_runtime> srt, std::vector<std::size_t> affine_indices) :
			mod(std::move(mod)), exec(exec), flame_size_reals(flame_size_reals), srt(srt), affine_indices(std::move(affine_indices)) {

			roccu::gpu_buffer<std::size_t> state_size_buf{ 1 };
			this->mod.kernel("get_sample_state_size").launch(1, 1)(state_size_buf.ptr());
			saved_state_size = state_size_buf.to_host()[0] * exec.first;
		}

		std::size_t saved_state_size;
		std::size_t flame_size_reals = 0;
		ezrtc::cuda_module mod = {};
		std::pair<std::uint32_t, std::uint32_t> exec = {};
		std::vector<std::size_t> affine_indices = {};

		std::shared_ptr<shared_runtime> srt;

	};

	static_assert(std::move_constructible<flame_kernel>);

	class flame_compiler: public traits::noncopyable, public traits::hashable {
	public:

		struct result: public traits::noncopyable {
			std::optional<flame_kernel> kernel = std::nullopt;
			std::string source = {};
			std::string log = {};
			double compile_ms = 0;

			result(std::string&& src, std::string&& log) noexcept : source(std::move(src)), log(std::move(log)) {}

			result(result&& o) noexcept {
				kernel = std::move(o.kernel);
				std::swap(source, o.source);
				std::swap(log, o.log);
			}
			result& operator=(result&& o) noexcept {
				kernel = std::move(o.kernel);
				std::swap(source, o.source);
				std::swap(log, o.log);

				return *this;
			}
		};

		void add_to_hash(rfkt::hash::state_t& state);

		static_assert(std::move_constructible<result>);

		auto get_flame_kernel(const flamedb& fdb, precision prec, const flame& f)-> result;
		std::string make_source(const flamedb& fdb, const rfkt::flame& f);

		explicit flame_compiler(std::shared_ptr<ezrtc::compiler> k_manager);

		flame_compiler& operator=(flame_compiler&& o) noexcept {
			*this = std::move(o);
			return *this;
		}

		flame_compiler(flame_compiler&& o) noexcept : km(o.km) {
			std::swap(shuf_bufs, o.shuf_bufs);
			std::swap(exec_configs, o.exec_configs);
			std::swap(num_shufs, o.num_shufs);
			std::swap(srt, o.srt);
			std::swap(compiled_variations, o.compiled_variations);
			std::swap(compiled_common, o.compiled_common);
			std::swap(last_flamedb_hash, o.last_flamedb_hash);
			std::swap(required_smem, o.required_smem);
			std::swap(km, o.km);
		}

	private:

		auto smem_per_block(precision prec, std::size_t flame_real_count, std::size_t threads_per_block) {
			auto sample_bytes = required_smem[std::make_pair(prec, threads_per_block)] + flame_real_count * (prec == precision::f32 ? 4: 8);
			sample_bytes += (sample_bytes % 16 > 0) ? 16 - sample_bytes % 16 : 0;
			return sample_bytes + iteration_info_size;
		}

		std::pair<roccu::execution_config, ezrtc::spec> make_opts(precision prec, const flame& f);

		std::shared_ptr<ezrtc::compiler> km;
		std::map<std::size_t, roccu::gpu_buffer<unsigned short>> shuf_bufs;
		decltype(std::declval<roccu::device_t>().concurrent_block_configurations()) exec_configs;

		std::map<std::pair<precision, std::size_t>, std::size_t> required_smem;
		std::size_t iteration_info_size = 0;

		std::size_t num_shufs = 4096;

		std::map<std::string, std::pair<std::string, std::string>, std::less<>> compiled_variations;
		std::map<std::string, std::string, std::less<>> compiled_common;

		rfkt::hash_t last_flamedb_hash = {};

		std::shared_ptr<flame_kernel::shared_runtime> srt;
	};
}