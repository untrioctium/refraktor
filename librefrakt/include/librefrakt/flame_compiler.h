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

			cuda_buffer<std::uint64_t> samples = {};

			saved_state() = delete;
			saved_state(saved_state&& o) noexcept {
				(*this) = std::move(o);
			}
			saved_state& operator=(saved_state&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(bin_dims, o.bin_dims);
				std::swap(shared, o.shared);
				std::swap(samples, o.samples);
				return *this;
			}

			saved_state(uint2 dims, std::size_t nbytes, std::size_t sample_buffer_size, CUstream stream) :
				bins(cuda::make_buffer_async<float4>(dims.x* dims.y, stream)),
				bin_dims(dims),
				shared(cuda::make_buffer_async<unsigned char>(nbytes, stream)),
				samples(cuda::make_buffer_async<std::uint64_t>(sample_buffer_size, stream)) {}
		};

		auto bin(CUstream stream, flame_kernel::saved_state& state, float target_quality, std::uint32_t ms_bailout, std::uint32_t iter_bailout) const->bin_result;
		
		auto warmup(CUstream stream, const flame& f, uint2 dims, double t, std::uint32_t nseg, double loops_per_frame, std::uint32_t seed, std::uint32_t count) const -> flame_kernel::saved_state
		{
			//auto warmup_impl = [&samples, &stream, &dims, seed, count, this]<typename Real>() -> flame_kernel::saved_state{
			using Real = float;
				const auto nreals = f.real_count() + 256 * 3;
				const auto pack_size_reals = nreals * (nseg + 3);
				const auto pack_size_bytes = sizeof(Real) * pack_size_reals;

				auto pack = new std::vector<Real>(pack_size_reals);
				auto packer = [pack, counter = 0](double v) mutable {
					(*pack)[counter] = static_cast<Real>(v);
					counter++;
				};

				const auto seg_length = loops_per_frame / nseg;
				for (int pos = -1; pos < static_cast<int>(nseg) + 2; pos++) {
					pack_flame(f, dims, packer, t + pos * seg_length);
				}

				CUdeviceptr samples_dev;
				cuMemAllocAsync(&samples_dev, pack_size_bytes, stream);
				cuMemcpyHtoDAsync(samples_dev, pack->data(), pack_size_bytes, stream);
				cuLaunchHostFunc(stream, [](void* ptr) {
					delete static_cast<decltype(pack)>(ptr);
				}, pack);

				CUdeviceptr segments_dev;
				cuMemAllocAsync(&segments_dev, pack_size_bytes * 4 * nseg, stream);
				auto [grid, block] = this->catmull->kernel().suggested_dims();
				auto nblocks = (nseg * pack_size_reals) / block;
				if ((nseg * pack_size_reals) % block > 0) nblocks++;
				this->catmull->kernel().launch(nblocks, block, stream, false)(
					samples_dev, static_cast<std::uint32_t>(nreals), std::uint32_t{ nseg }, segments_dev
				);
				cuMemFreeAsync(samples_dev, stream);

				auto state = flame_kernel::saved_state{ dims, this->saved_state_size(), 1000, stream };

				cuMemsetD32Async(state.bins.ptr(), 0, state.bin_dims.x * state.bin_dims.y * 4, stream);

				this->mod.kernel("warmup")
					.launch(this->exec.first, this->exec.second, stream, true)
					(
						std::uint32_t{ nseg },
						segments_dev,
						this->shuf_bufs->ptr(),
						seed, count,
						state.shared.ptr()
					);

				cuMemFreeAsync(segments_dev, stream);
				return state;
			//};
		
			//if (this->prec == precision::f32) return warmup_impl.operator()<float>();
			//else return warmup_impl.operator()<double> ();
		}

	private:

		static void pack_flame(const flame& f, uint2 dims, auto& packer, double t) {
			auto mat = f.make_screen_space_affine(dims.x, dims.y, t);
			packer(mat.a.sample(t)); 
			packer(mat.d.sample(t));
			packer(mat.b.sample(t));
			packer(mat.e.sample(t));
			packer(mat.c.sample(t));
			packer(mat.f.sample(t));

			auto sum = 0.0;
			for (const auto& xf : f.xforms) sum += xf.weight.sample(t);
			packer(sum);

			auto pack_xform = [&packer, &t](const xform& xf) {
				packer(xf.weight.sample(t));
				packer(xf.color.sample(t));
				packer(xf.color_speed.sample(t));
				packer(xf.opacity.sample(t));

				for (const auto& vl : xf.vchain) {

					auto affine = vl.affine.scale(vl.add_mod_scale.sample(t)).rotate(vl.aff_mod_rotate.sample(t)).translate(vl.aff_mod_translate.first.sample(t), vl.aff_mod_translate.second.sample(t));

					packer(affine.a.sample(t));
					packer(affine.d.sample(t));
					packer(affine.b.sample(t));
					packer(affine.e.sample(t));
					packer(affine.c.sample(t));
					packer(affine.f.sample(t));

					for (const auto& [idx, value] : vl.variations) packer(value.sample(t));
					for (const auto& [idx, value] : vl.parameters) packer(value.sample(t));
				}
			};

			for (const auto& xf : f.xforms) pack_xform(xf);
			if (f.final_xform.has_value()) pack_xform(*f.final_xform);

			for (const auto& hsv : f.palette()) {
				packer(hsv[0].sample(t));
				packer(hsv[1].sample(t));
				packer(hsv[2].sample(t));
			}
		}

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