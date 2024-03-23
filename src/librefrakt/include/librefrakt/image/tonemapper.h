#include <ezrtc.h>
#include <librefrakt/gpu_buffer.h>
#include <librefrakt/util/cuda.h>

namespace rfkt {
	class tonemapper {
	public:
		tonemapper(ezrtc::compiler& km);

		struct args_t {
			double quality;
			double gamma;
			double brightness;
			double vibrancy;
		};

		void run(roccu::gpu_span<float4> bins, roccu::gpu_span<half3> out, const args_t& args, roccu::gpu_stream& stream) const;

	private:
		ezrtc::cuda_module tm;
		int block_size;
	};
}