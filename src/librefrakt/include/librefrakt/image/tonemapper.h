#include <ezrtc.h>
#include <librefrakt/cuda_buffer.h>

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

		void run(cuda_span<float4> bins, cuda_span<half3> out, const args_t& args, cuda_stream& stream) const;

	private:
		ezrtc::cuda_module tm;
		int block_size;
	};
}