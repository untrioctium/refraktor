#pragma once

#include <ezrtc.hpp>
#include <roccu_cpp_types.hpp>

#include <librefrakt/vector_types.hpp>

namespace rfkt {
	class tonemapper {
	public:
		tonemapper(ezrtc::compiler& km);

		struct args_t {
			double quality;
			double gamma;
			double brightness;
			double vibrancy;
			bool hdr;
		};

		void run(roccu::gpu_image_view<float4> cold_bins, roccu::gpu_image_view<half4> hot_bins, roccu::gpu_image_view<half3> out, const args_t& args, roccu::gpu_stream& stream) const;
		void run(roccu::gpu_image_view<float4> cold_bins, roccu::gpu_image_view<half4> hot_bins, roccu::gpu_image_view<half4> out, const args_t& args, roccu::gpu_stream& stream) const;

		void run(roccu::gpu_image_view<float4> cold_bins, roccu::gpu_image_view<half4> hot_bins, roccu::gpu_image_view<float3> out, const args_t& args, roccu::gpu_stream& stream) const;
		void run(roccu::gpu_image_view<float4> cold_bins, roccu::gpu_image_view<half4> hot_bins, roccu::gpu_image_view<float4> out, const args_t& args, roccu::gpu_stream& stream) const;

	private:

		void run_impl(const std::string& kernel, CUdeviceptr cold_bins, CUdeviceptr hot_bins, CUdeviceptr out, unsigned int width, unsigned int size, unsigned int scaling, const args_t& args, roccu::gpu_stream& stream) const;

		ezrtc::cuda_module tm;
		int block_size;
	};
}