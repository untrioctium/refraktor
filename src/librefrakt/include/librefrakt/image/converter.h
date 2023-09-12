#include <ezrtc.h>
#include <spdlog/spdlog.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/cuda_buffer.h>

namespace rfkt {
	class converter {
	public:
		explicit converter(ezrtc::compiler& c) {
			auto conv_result = c.compile(
				ezrtc::spec::source_file("convert", "assets/kernels/convert.cu")
				.kernel("convert<true>")
				.kernel("convert<false>")
				.kernel("to_float3")
				.flag(ezrtc::compile_flag::extra_device_vectorization)
				.flag(ezrtc::compile_flag::use_fast_math)
				.flag(ezrtc::compile_flag::default_device)
			);

			if (not conv_result.module.has_value()) {
				SPDLOG_CRITICAL("{}", conv_result.log);
				exit(1);
			}

			conv = std::move(conv_result.module.value());

			auto [s_grid, s_block] = conv.kernel("convert<false>").suggested_dims();
			auto [s_grid_planar, s_block_planar] = conv.kernel("convert<true>").suggested_dims();

			block_size = s_block;
			block_size_planar = s_block_planar;
		}

		void to_24bit(cuda_span<half3> in, cuda_span<uchar4> out, bool planar, cuda_stream& stream) const {
			auto kernel = (planar) ? conv.kernel("convert<true>") : conv.kernel("convert<false>");

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

		void to_float3(cuda_span<half3> in, cuda_span<float4> out, cuda_stream& stream) const {
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

	private:
		ezrtc::cuda_module conv;
		int block_size;
		int block_size_planar;
	};
}