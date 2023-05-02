#pragma once

#include <typeinfo>
#include <future>

#include <ezrtc.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_types.h>
#include <librefrakt/cuda_buffer.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/hash.h>

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
			cuda_buffer<float4> bins = {};
			uint2 bin_dims = {};
			double quality = 0.0;
			cuda_buffer<> shared = {};

			saved_state() = default;
			saved_state(saved_state&& o) noexcept {
				(*this) = std::move(o);
			}
			saved_state& operator=(saved_state&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(bin_dims, o.bin_dims);
				std::swap(shared, o.shared);
				std::swap(quality, o.quality);
				return *this;
			}

			saved_state(uint2 dims, std::size_t nbytes, CUstream stream) :
				bins(dims.x* dims.y, stream),
				bin_dims(dims),
				shared(nbytes, stream) {
				bins.clear(stream);
			}
		};

		struct bailout_args {
			std::uint32_t iters = 1000000;
			std::uint32_t millis = 1000;
			double quality = 128.0;
		};

		auto bin(cuda_stream& stream, flame_kernel::saved_state& state, const bailout_args&) const-> std::future<bin_result>;
		auto warmup(cuda_stream& stream, std::span<const flame> samples, uint2 dims, std::uint32_t seed, std::uint32_t count) const->flame_kernel::saved_state;

		flame_kernel(flame_kernel&& o) noexcept {
			*this = std::move(o);
		}

		flame_kernel& operator=(flame_kernel&& o) noexcept {
			this->flame_hash = o.flame_hash;
			this->exec = o.exec;
			this->mod = std::move(o.mod);
			this->catmull = std::move(o.catmull);
			this->device_hz = o.device_hz;
			this->flame_size_reals = o.flame_size_reals;

			return *this;
		}

		flame_kernel() = default;

	private:

		std::size_t saved_state_size() const { return mod("bin").shared_bytes() * exec.first; }

		flame_kernel(const rfkt::hash_t& flame_hash, std::size_t flame_size_reals, ezrtc::cuda_module&& mod, std::pair<int, int> exec, std::shared_ptr<ezrtc::cuda_module>& catmull, std::size_t device_hz) :
			mod(std::move(mod)), flame_hash(flame_hash), exec(exec), catmull(catmull), device_hz(device_hz), flame_size_reals(flame_size_reals) {
		}

		std::shared_ptr<ezrtc::cuda_module> catmull = nullptr;
		std::size_t flame_size_reals = 0;
		rfkt::hash_t flame_hash = {};
		ezrtc::cuda_module mod = {};
		std::pair<std::uint32_t, std::uint32_t> exec = {};
		std::size_t device_hz;
	};

	static_assert(std::move_constructible<flame_kernel>);

	class flame_compiler: public traits::noncopyable, public traits::hashable {
	public:

		struct result: public traits::noncopyable {
			std::optional<flame_kernel> kernel = std::nullopt;
			std::string source = {};
			std::string log = {};

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

		flame_compiler(ezrtc::compiler& k_manager);
	private:

		auto smem_per_block(precision prec, int flame_real_bytes, int threads_per_block) {
			const auto per_thread_size = (prec == precision::f32) ? 25 : 48;
			return per_thread_size * threads_per_block + flame_real_bytes + 820;
		}

		std::pair<cuda::execution_config, ezrtc::spec> make_opts(precision prec, const flame& f);

		ezrtc::compiler& km;
		std::map<std::size_t, cuda_buffer<unsigned short>> shuf_bufs;
		decltype(std::declval<cuda::device_t>().concurrent_block_configurations()) exec_configs;
		std::size_t num_shufs = 4096;

		std::shared_ptr<ezrtc::cuda_module> catmull;

		std::map<std::string, std::pair<std::string, std::string>, std::less<>> compiled_variations;
		std::map<std::string, std::string, std::less<>> compiled_common;

		rfkt::hash_t last_flamedb_hash = {};
	};
}