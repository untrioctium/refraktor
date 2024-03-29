#include <spdlog/spdlog.h>

#include <librefrakt/image/tonemapper.h>

#define RFKT_ASSERT(x) 

namespace rfkt {

	tonemapper::tonemapper(ezrtc::compiler& km) {
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
		block_size = s_block;
	}

	void tonemapper::run(roccu::gpu_span<float4> bins, roccu::gpu_span<half3> out, const args_t& args, roccu::gpu_stream& stream) const {

		unsigned int size = bins.size();
		auto nblocks = size / block_size;
		if (size % block_size != 0) {
			nblocks++;
		}

		CUDA_SAFE_CALL(tm.kernel("tonemap").launch(nblocks, block_size, stream)(
			bins.ptr(),
			out,
			size,
			static_cast<float>(args.gamma),
			std::powf(10.0f, -log10f(args.quality) - 0.5f),
			static_cast<float>(args.brightness),
			static_cast<float>(args.vibrancy)
			));

	}

}