#include <ezrtc.h>
#include <spdlog/spdlog.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/cuda_buffer.h>

namespace rfkt {
	class tonemapper {
	public:
		tonemapper(ezrtc::compiler& km) {
			auto tm_result = km.compile(
				ezrtc::spec::source_file("tonemap", "assets/kernels/tonemap.cu")
				.kernel("tonemap")
				.flag(ezrtc::compile_flag::extra_device_vectorization)
				.flag(ezrtc::compile_flag::use_fast_math)
				.flag(ezrtc::compile_flag::default_device)
			);

			if (!tm_result.module.has_value()) {
				SPDLOG_ERROR("{}", tm_result.log);
				exit(1);
			}

			auto func = tm_result.module->kernel();
			auto [s_grid, s_block] = func.suggested_dims();
			SPDLOG_INFO("Loaded tonemapper kernel: {} regs, {} shared, {} local, {}x{} suggested dims", func.register_count(), func.shared_bytes(), func.local_bytes(), s_grid, s_block);

			tm = std::move(tm_result.module.value());
		}

		void run(cuda_view<float4> bins, cuda_view<half3> out, uint2 dims, double quality, double gamma, double brightness, double vibrancy, cuda_stream& stream) const {

			CUDA_SAFE_CALL(tm.kernel("tonemap").launch({ dims.x / 8 + 1, dims.y / 8 + 1, 1 }, { 8, 8, 1 }, stream)(
				bins.ptr(),
				0,
				out,
				dims.x, dims.y,
				static_cast<float>(gamma),
				std::powf(10.0f, -log10f(quality) - 0.5f),
				static_cast<float>(brightness),
				static_cast<float>(vibrancy)
				));

		}

		void run(cuda_view<float4> bins, cuda_view<ushort4> accumulator, cuda_view<half3> out, uint2 dims, double quality, double gamma, double brightness, double vibrancy, cuda_stream& stream) const {

			CUDA_SAFE_CALL(tm.kernel("tonemap").launch({ dims.x / 8 + 1, dims.y / 8 + 1, 1 }, { 8, 8, 1 }, stream)(
				bins.ptr(),
				accumulator.ptr(),
				out,
				dims.x, dims.y,
				static_cast<float>(gamma),
				std::powf(10.0f, -log10f(quality) - 0.5f),
				static_cast<float>(brightness),
				static_cast<float>(vibrancy)
				));

		}
	private:
		ezrtc::module tm;
	};
}