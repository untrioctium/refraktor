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
				.kernel("convert_32<true>")
				.kernel("convert_32<false>")
				.kernel("convert_24")
				.kernel("to_float3")
				.kernel("to_half4")
				.flag(ezrtc::compile_flag::extra_device_vectorization)
				.flag(ezrtc::compile_flag::use_fast_math)
				.flag(ezrtc::compile_flag::default_device)
			);

			if (not conv_result.module.has_value()) {
				SPDLOG_CRITICAL("{}", conv_result.log);
				exit(1);
			}

			conv = std::move(conv_result.module.value());

			auto [s_grid, s_block] = conv.kernel("convert_32<false>").suggested_dims();
			auto [s_grid_planar, s_block_planar] = conv.kernel("convert_32<true>").suggested_dims();

			block_size = s_block;
			block_size_planar = s_block_planar;
		}

		void to_32bit(roccu::gpu_span<half3> in, roccu::gpu_span<uchar4> out, bool planar, roccu::gpu_stream& stream) const {
			auto kernel = (planar) ? conv.kernel("convert_32<true>") : conv.kernel("convert_32<false>");

			auto b_size = (planar) ? block_size_planar : block_size;
			unsigned int size = in.size();
			auto nblocks = size / b_size;
			if (size % b_size != 0) {
				nblocks++;
			}

			CUDA_SAFE_CALL(kernel
				.launch(nblocks, b_size, stream)
				(
					in.ptr(),
					out.ptr(),
					size
				));
		}

		void to_24bit(roccu::gpu_span<half3> in, roccu::gpu_span<uchar3> out, roccu::gpu_stream& stream) const {
			unsigned int size = in.size();
			auto nblocks = size / block_size;
			if (size % block_size != 0) {
				nblocks++;
			}

			CUDA_SAFE_CALL(conv.kernel("convert_24")
				.launch(nblocks, block_size, stream)
				(
					in.ptr(),
					out.ptr(),
					size
				));
		}

		void to_float3(roccu::gpu_span<half3> in, roccu::gpu_span<float4> out, roccu::gpu_stream& stream) const {
			unsigned int size = in.size();
			auto nblocks = size / block_size;
			if (size % block_size != 0) {
				nblocks++;
			}

			CUDA_SAFE_CALL(conv.kernel("to_float3")
				.launch(nblocks, block_size, stream)
				(
					in.ptr(),
					out.ptr(),
					size
					));
		}

		void to_half4(roccu::gpu_span<half3> in, roccu::gpu_span<half4> out, roccu::gpu_stream& stream) const {
			unsigned int size = in.size();
			auto nblocks = size / block_size;
			if (size % block_size != 0) {
				nblocks++;
			}

			CUDA_SAFE_CALL(conv.kernel("to_half4")
				.launch(nblocks, block_size, stream)
				(
					in.ptr(),
					out.ptr(),
					size
					));
		}

	private:
		ezrtc::cuda_module conv;
		int block_size;
		int block_size_planar;
	};
}