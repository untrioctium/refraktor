#pragma once

#include <typeinfo>

#include <librefrakt/kernel_manager.h>	
#include <librefrakt/flame_types.h>
#include <librefrakt/cuda_buffer.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/hash.h>

namespace rfkt {

	class flame_compiler;

	class flame_kernel {

	public:
		friend class flame_compiler;

		struct bin_result {
			float quality;
			float elapsed_ms;
			std::size_t total_passes;
			std::size_t total_draws;
			std::size_t total_bins;
			double passes_per_thread;
		};

		struct saved_state: traits::noncopyable<saved_state> {
			cuda_buffer<float4> bins = {};
			uint2 bin_dims = {};
			cuda_buffer<> shared = {};

			saved_state() = delete;
			saved_state(saved_state&& o) noexcept {
				(*this) = std::move(o);
			}
			saved_state& operator=(saved_state&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(bin_dims, o.bin_dims);
				std::swap(shared, o.shared);
				return *this;
			}

			saved_state(uint2 dims, std::size_t nbytes, CUstream stream) :
				bins(cuda::make_buffer_async<float4>(dims.x* dims.y, stream)),
				bin_dims(dims),
				shared(cuda::make_buffer_async<unsigned char>(nbytes, stream))
				/*samples(cuda::make_buffer_async<std::uint64_t>(sample_buffer_size, stream))*/ {}
		};

		auto bin(CUstream stream, flame_kernel::saved_state& state, float target_quality, std::uint32_t ms_bailout, std::uint32_t iter_bailout) const->bin_result;
		auto warmup(CUstream stream, const flame& f, uint2 dims, double t, std::uint32_t nseg, double loops_per_frame, std::uint32_t seed, std::uint32_t count) const->flame_kernel::saved_state;

	private:

		std::size_t saved_state_size() const { return mod("bin").shared_bytes() * exec.first; }

		flame_kernel(const rfkt::hash_t& flame_hash, cuda_module&& mod, precision prec, std::shared_ptr<cuda_buffer<unsigned short>> shuf, long long mhz, std::pair<int,int> exec, std::shared_ptr<cuda_module> catmull) :
			mod(std::move(mod)), flame_hash(flame_hash), prec(prec), shuf_bufs(shuf), device_mhz(mhz), exec(exec), catmull(catmull) {
		}

		precision prec;
		rfkt::hash_t flame_hash;
		long long device_mhz;
		std::pair<std::uint32_t, std::uint32_t> exec;

		cuda_module mod;

		std::shared_ptr<cuda_module> catmull;
		std::shared_ptr<cuda_buffer<unsigned short>> shuf_bufs;
	};

	class flame_compiler {
	public:

		struct result {
			std::optional<flame_kernel> kernel;
			std::string source;
			std::string log;
		};

		auto get_flame_kernel(precision prec, const flame* f)-> result;
		bool is_cached(precision prec, const flame* f);

		flame_compiler(kernel_manager& k_manager);
	private:

		auto smem_per_block(precision prec, int flame_real_bytes, int threads_per_block) {
			const auto per_thread_size = (prec == precision::f32) ? 25 : 48;
			return per_thread_size * threads_per_block + flame_real_bytes + 820;
		}

		std::pair<cuda::execution_config, compile_opts> make_opts(precision prec, const flame* f);

		kernel_manager& km;
		std::map<std::size_t, std::shared_ptr<cuda_buffer<unsigned short>>> shuf_bufs;
		decltype(std::declval<cuda::device_t>().concurrent_block_configurations()) exec_configs;
		std::size_t num_shufs = 4096;
		long long device_mhz;

		std::shared_ptr<cuda_module> catmull;
	};
}