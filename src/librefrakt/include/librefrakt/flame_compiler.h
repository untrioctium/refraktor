#pragma once

#include <typeinfo>
#include <future>

#include <ezrtc.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_types.h>
#include <librefrakt/cuda_buffer.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/hash.h>

#include <librefrakt/allocators.h>

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
		};

		struct saved_state: public traits::noncopyable {
			cuda_image<float4> bins = {};
			double quality = 0.0;
			cuda_buffer<> shared = {};
			cuda_buffer<bool> stopper;

			saved_state() = default;
			saved_state(saved_state&& o) noexcept {
				(*this) = std::move(o);
			}
			saved_state& operator=(saved_state&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(shared, o.shared);
				std::swap(quality, o.quality);
				std::swap(stopper, o.stopper);
				return *this;
			}

			saved_state(uint2 dims, std::size_t nbytes, CUstream stream) :
				bins(dims.x, dims.y, stream),
				shared(nbytes, stream),
				stopper(1, stream) {
				bins.clear(stream);
			}

			saved_state(decltype(saved_state::bins)&& bins, std::size_t nbytes, CUstream stream) :
				bins(std::move(bins)),
				shared(nbytes, stream),
				stopper(1, stream) {
				bins.clear(stream);
			}

			void abort_binning() {
				const static bool stop = true;
				stopper.from_host({ &stop, 1 }, nullptr);
			}

		};

		struct bailout_args {
			std::uint32_t iters = 4'000'000'000;
			std::uint32_t millis = 1000;
			double quality = 128.0;
		};

		auto bin(cuda_stream& stream, flame_kernel::saved_state& state, const bailout_args&) const-> std::future<bin_result>;
		auto warmup(cuda_stream& stream, std::span<double> samples, uint2 dims, std::uint32_t seed, std::uint32_t count) const->flame_kernel::saved_state;
		auto warmup(cuda_stream& stream, std::span<double> samples, cuda_image<float4>&& bins, std::uint32_t seed, std::uint32_t count) const->flame_kernel::saved_state;

		flame_kernel(flame_kernel&& o) noexcept {
			*this = std::move(o);
		}

		flame_kernel& operator=(flame_kernel&& o) noexcept {
			this->exec = o.exec;
			this->mod = std::move(o.mod);
			this->flame_size_reals = o.flame_size_reals;
			this->srt = std::move(o.srt);
			this->affine_indices = std::move(o.affine_indices);

			return *this;
		}

		flame_kernel() = default;

		bool valid() const {
			return mod.operator bool();
		}

	private:

		struct shared_runtime {
			ezrtc::cuda_module catmull = {};
			rfkt::pinned_ring_allocator pra;
			rfkt::device_ring_allocator dra;
		};

		std::size_t saved_state_size() const { return mod("bin").shared_bytes() * exec.first; }

		flame_kernel(std::size_t flame_size_reals, ezrtc::cuda_module&& mod, std::pair<int, int> exec, std::shared_ptr<shared_runtime> srt, std::vector<std::size_t> affine_indices) :
			mod(std::move(mod)), exec(exec), flame_size_reals(flame_size_reals), srt(srt), affine_indices(std::move(affine_indices)) {
		}

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
		}

	private:

		auto smem_per_block(precision prec, int flame_real_bytes, int threads_per_block) {
			return required_smem[std::make_pair(prec, threads_per_block)] + flame_real_bytes;
		}

		std::pair<cuda::execution_config, ezrtc::spec> make_opts(precision prec, const flame& f);

		std::shared_ptr<ezrtc::compiler> km;
		std::map<std::size_t, cuda_buffer<unsigned short>> shuf_bufs;
		decltype(std::declval<cuda::device_t>().concurrent_block_configurations()) exec_configs;

		std::map<std::pair<precision, int>, int> required_smem;

		std::size_t num_shufs = 4096;

		std::map<std::string, std::pair<std::string, std::string>, std::less<>> compiled_variations;
		std::map<std::string, std::string, std::less<>> compiled_common;

		rfkt::hash_t last_flamedb_hash = {};

		std::shared_ptr<flame_kernel::shared_runtime> srt;
	};
}