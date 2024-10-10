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
			double max_density;
		};

		void run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<half3> out, const args_t& args, roccu::gpu_stream& stream) const;
		void run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<half4> out, const args_t& args, roccu::gpu_stream& stream) const;

		void run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<float3> out, const args_t& args, roccu::gpu_stream& stream) const;
		void run(roccu::gpu_image_view<float4> bins, roccu::gpu_image_view<float4> out, const args_t& args, roccu::gpu_stream& stream) const;

		void density_hdr(roccu::gpu_image_view<half4> in, roccu::gpu_image_view<float4> out, roccu::gpu_span<std::size_t> densities, roccu::gpu_stream& stream) const;
		void density_hdr(roccu::gpu_image_view<float4> in, roccu::gpu_image_view<float4> out, roccu::gpu_span<std::size_t> densities, roccu::gpu_stream& stream) const;

	private:

		void run_impl(const std::string& kernel, RUdeviceptr bins, RUdeviceptr out, unsigned int size, const args_t& args, roccu::gpu_stream& stream) const;
		void density_hdr_impl(const std::string& kernel, RUdeviceptr in, RUdeviceptr out, unsigned int size, RUdeviceptr densities, roccu::gpu_stream& stream) const;

		ezrtc::cuda_module tm;
		int block_size;
	};
}