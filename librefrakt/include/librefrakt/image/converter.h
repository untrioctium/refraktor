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
				.flag(ezrtc::compile_flag::extra_device_vectorization)
				.flag(ezrtc::compile_flag::use_fast_math)
				.flag(ezrtc::compile_flag::default_device)
			);

			if (not conv_result.module.has_value()) {
				SPDLOG_CRITICAL("{}", conv_result.log);
				exit(1);
			}

			conv = std::move(conv_result.module.value());
		}

		void to_24bit(cuda_view<half3> in, cuda_view<uchar4> out, uint2 dims, bool planar, cuda_stream& stream) const {
			auto kernel = (planar) ? conv.kernel("convert<true>") : conv.kernel("convert<false>");

			CUDA_SAFE_CALL(kernel
				.launch({ dims.x / 8 + 1, dims.y / 8 + 1, 1 }, { 8, 8, 1 }, stream)
				(
					in.ptr(),
					out.ptr(),
					dims.x,
					dims.y
				));
		}

	private:
		ezrtc::cuda_module conv;
	};
}