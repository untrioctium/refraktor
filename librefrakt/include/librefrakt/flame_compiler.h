#pragma once

#include <typeinfo>
#include <future>

#include <librefrakt/kernel_manager.h>	
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
			float quality;
			float elapsed_ms;
			std::size_t total_passes;
			std::size_t total_draws;
			std::size_t total_bins;
			double passes_per_thread;
			std::vector<double> xform_selections;
		};

		struct saved_state: public traits::noncopyable {
			cuda_buffer<float4> bins = {};
			uint2 bin_dims = {};
			std::vector<double> norm_xform_weights;
			cuda_buffer<> shared = {};

			double warmup_ms = 0.0;

			saved_state() = delete;
			saved_state(saved_state&& o) noexcept {
				(*this) = std::move(o);
			}
			saved_state& operator=(saved_state&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(bin_dims, o.bin_dims);
				std::swap(shared, o.shared);
				std::swap(norm_xform_weights, o.norm_xform_weights);
				return *this;
			}

			saved_state(uint2 dims, std::size_t nbytes, CUstream stream, std::vector<double>&& norm_xform_weights) :
				bins(cuda::make_buffer_async<float4>(dims.x* dims.y, stream)),
				bin_dims(dims),
				shared(cuda::make_buffer_async<unsigned char>(nbytes, stream)),
				norm_xform_weights(std::move(norm_xform_weights))
				/*samples(cuda::make_buffer_async<std::uint64_t>(sample_buffer_size, stream))*/ {}
		};

		auto bin(CUstream stream, flame_kernel::saved_state& state, float target_quality, std::uint32_t ms_bailout, std::uint32_t iter_bailout) const-> std::future<bin_result>;
		auto warmup(CUstream stream, const flame& f, uint2 dims, double t, std::uint32_t nseg, double loops_per_frame, std::uint32_t seed, std::uint32_t count) const->flame_kernel::saved_state;

		flame_kernel(flame_kernel&& o) {
			*this = std::move(o);
		}

		flame_kernel& operator=(flame_kernel&& o) {
			this->prec = o.prec;
			this->flame_hash = o.flame_hash;
			this->device_mhz = o.device_mhz;
			this->exec = o.exec;

			this->mod = std::move(o.mod);
			this->catmull = std::move(o.catmull);
			this->shuf_bufs = std::move(o.shuf_bufs);

			return *this;
		}

		flame_kernel() {}

	private:

		std::size_t saved_state_size() const { return mod("bin").shared_bytes() * exec.first; }

		flame_kernel(const rfkt::hash_t& flame_hash, cuda_module&& mod, precision prec, std::shared_ptr<cuda_buffer<unsigned short>> shuf, long long mhz, std::pair<int,int> exec, std::shared_ptr<cuda_module> catmull) :
			mod(std::move(mod)), flame_hash(flame_hash), prec(prec), shuf_bufs(shuf), device_mhz(mhz), exec(exec), catmull(catmull) {
		}

		precision prec = rfkt::precision::f32;
		rfkt::hash_t flame_hash = {};
		long long device_mhz = 0;
		std::pair<std::uint32_t, std::uint32_t> exec = {};

		cuda_module mod = {};

		std::shared_ptr<cuda_module> catmull = nullptr;
		std::shared_ptr<cuda_buffer<unsigned short>> shuf_bufs = nullptr;
	};

	static_assert(std::move_constructible<flame_kernel>);

	class flame_compiler {
	public:

		struct result: public traits::noncopyable {
			std::optional<flame_kernel> kernel = std::nullopt;
			std::string source = {};
			std::string log = {};

			result(std::string&& src, std::string&& log) : source(std::move(src)), log(std::move(log)), kernel(std::nullopt) {}

			result(result&& o) {
				kernel = std::move(o.kernel);
				std::swap(source, o.source);
				std::swap(log, o.log);
			}
			result& operator=(result&& o) {
				kernel = std::move(o.kernel);
				std::swap(source, o.source);
				std::swap(log, o.log);

				return *this;
			}
		};

		static_assert(std::move_constructible<result>);

		auto get_flame_kernel(precision prec, const flame& f)-> result;
		bool is_cached(precision prec, const flame& f);

		flame_compiler(kernel_manager& k_manager);
	private:

		auto smem_per_block(precision prec, int flame_real_bytes, int threads_per_block) {
			const auto per_thread_size = (prec == precision::f32) ? 25 : 48;
			return per_thread_size * threads_per_block + flame_real_bytes + 820;
		}

		std::pair<cuda::execution_config, compile_opts> make_opts(precision prec, const flame& f);

		kernel_manager& km;
		std::map<std::size_t, std::shared_ptr<cuda_buffer<unsigned short>>> shuf_bufs;
		decltype(std::declval<cuda::device_t>().concurrent_block_configurations()) exec_configs;
		std::size_t num_shufs = 4096;
		long long device_mhz;

		std::shared_ptr<cuda_module> catmull;
	};
}