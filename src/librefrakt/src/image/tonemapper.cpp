#include <spdlog/spdlog.h>

#include <librefrakt/image/tonemapper.h>
#include <librefrakt/constants.h>

#define RFKT_ASSERT(x) 

namespace rfkt {

	tonemapper::tonemapper(ezrtc::compiler& km) {
		auto tm_result = km.compile(
			ezrtc::spec::source_file("tonemap", "assets/kernels/tonemap.cu")
			.kernel(std::format("tonemap<half3, {}>", histogram_granularity))
			.kernel(std::format("tonemap<half4, {}>", histogram_granularity))
			.kernel(std::format("tonemap<float3, {}>", histogram_granularity))
			.kernel(std::format("tonemap<float4, {}>", histogram_granularity))
			.kernel(std::format("density_hdr<half4, {}>", histogram_granularity))
			.kernel(std::format("density_hdr<float4, {}>", histogram_granularity))
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

	void tonemapper::run_impl(const std::string& kernel, RUdeviceptr bins, RUdeviceptr out, unsigned int size, const args_t& args, roccu::gpu_stream& stream) const {
		auto nblocks = size / block_size;
		if (size % block_size != 0) {
			nblocks++;
		}

		SPDLOG_INFO("Quality: {} {} max, {}x average", args.quality, args.max_density, args.max_density / args.quality);

		CUDA_SAFE_CALL(tm.kernel(kernel).launch(nblocks, block_size, stream)(
			bins,
			out,
			size,
			args.quality,
			args.max_density,
			static_cast<float>(args.gamma),
			std::powf(10.0f, -log10f(args.quality) - 0.5f),
			static_cast<float>(args.brightness),
			static_cast<float>(args.vibrancy)
			));
	}

	void tonemapper::density_hdr_impl(const std::string& kernel, RUdeviceptr in, RUdeviceptr out, unsigned int size, RUdeviceptr densities, roccu::gpu_stream& stream) const {
		auto nblocks = size / block_size;
		if (size % block_size != 0) {
			nblocks++;
		}

		CUDA_SAFE_CALL(tm.kernel(kernel).launch(nblocks, block_size, stream)(
			in,
			out,
			size,
			densities
			));
	}

	void tonemapper::run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<half3> out, const args_t& args, roccu::gpu_stream& stream) const {
		run_impl(std::format("tonemap<half3, {}>", histogram_granularity), bins.ptr(), out.ptr(), bins.area(), args, stream);
	}

	void tonemapper::run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<half4> out, const args_t& args, roccu::gpu_stream& stream) const {
		run_impl(std::format("tonemap<half4, {}>", histogram_granularity), bins.ptr(), out.ptr(), bins.area(), args, stream);
	}

	void tonemapper::run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<float3> out, const args_t& args, roccu::gpu_stream& stream) const {
		run_impl(std::format("tonemap<float3, {}>", histogram_granularity), bins.ptr(), out.ptr(), bins.area(), args, stream);
	}

	void tonemapper::run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<float4> out, const args_t& args, roccu::gpu_stream& stream) const {
		run_impl(std::format("tonemap<float4, {}>", histogram_granularity), bins.ptr(), out.ptr(), bins.area(), args, stream);
	}

	void tonemapper::density_hdr(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<float4> out, roccu::gpu_span<std::size_t> densities, roccu::gpu_stream& stream) const {
		density_hdr_impl(std::format("density_hdr<half4, {}>", histogram_granularity), in.ptr(), out.ptr(), in.area(), densities.ptr(), stream);
	}

	void tonemapper::density_hdr(roccu::gpu_image_view<float4> in, roccu::gpu_image_view<float4> out, roccu::gpu_span<std::size_t> densities, roccu::gpu_stream& stream) const {
		density_hdr_impl(std::format("density_hdr<float4, {}>", histogram_granularity), in.ptr(), out.ptr(), in.area(), densities.ptr(), stream);
	}
}