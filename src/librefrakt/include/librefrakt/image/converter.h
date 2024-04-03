#include <ezrtc.h>
#include <spdlog/spdlog.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/gpu_buffer.h>

namespace rfkt {
	class converter {
	public:
		explicit converter(ezrtc::compiler& c) {
			auto conv_result = c.compile(
				ezrtc::spec::source_file("convert", "assets/kernels/convert.cu")
				.kernel("convert_to<half3, uchar3>")
				.kernel("convert_to<half3, uchar4>")
				.kernel("convert_to<half3, float4>")
				.kernel("convert_to<half3, float3>")
				.kernel("convert_to<half3, half4>")
				.kernel("convert_to<half4, half3>")
				.kernel("convert_to<half4, uchar3>")
				.kernel("convert_to<half4, uchar4>")
				.kernel("convert_to<half4, float4>")
				.kernel("convert_to<half4, float3>")
				.flag(ezrtc::compile_flag::extra_device_vectorization)
				.flag(ezrtc::compile_flag::use_fast_math)
				.flag(ezrtc::compile_flag::default_device)
			);

			if (not conv_result.module.has_value()) {
				SPDLOG_CRITICAL("{}", conv_result.log);
				exit(1);
			}

			conv = std::move(conv_result.module.value());

			auto [s_grid, s_block] = conv.kernel("convert_to<half3, uchar3>").suggested_dims();

			block_size = s_block;
		}

		void to_half4(roccu::gpu_image_view<half3> in, roccu::gpu_image_view<half4> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half3, half4>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_float3(roccu::gpu_image_view<half3> in, roccu::gpu_image_view<float3> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half3, float3>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_float4(roccu::gpu_image_view<half3> in, roccu::gpu_image_view<float4> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half3, float4>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_uchar3(roccu::gpu_image_view<half3> in, roccu::gpu_image_view<uchar3> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half3, uchar3>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_uchar4(roccu::gpu_image_view<half3> in, roccu::gpu_image_view<uchar4> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half3, uchar4>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_half3(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<half3> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half4, half3>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_uchar3(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<uchar3> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half4, uchar3>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_uchar4(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<uchar4> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half4, uchar4>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_float4(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<float4> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half4, float4>", in.ptr(), out.ptr(), in.area(), stream);
		}

		void to_float3(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<float3> out, roccu::gpu_stream& stream) {
			run_convert("convert_to<half3, float3>", in.ptr(), out.ptr(), in.area(), stream);
		}

	private:

		void run_convert(const std::string& name, RUdeviceptr in, RUdeviceptr out, unsigned int size, roccu::gpu_stream& stream) const {
			auto nblocks = size / block_size;
			if (size % block_size != 0) {
				nblocks++;
			}

			CUDA_SAFE_CALL(conv.kernel(name)
				.launch(nblocks, block_size, stream)
				(
					in,
					out,
					size
				));
		}

		ezrtc::cuda_module conv;
		int block_size;
	};
}