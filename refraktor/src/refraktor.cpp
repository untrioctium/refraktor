#include <iostream>
#include <fmt/format.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util.h>
#include <librefrakt/util/stb.h>
#include <librefrakt/util/filesystem.h>
#include <signal.h>

bool break_loop = false;

int main() {

	rfkt::flame_info::initialize("config/variations.yml");
	auto ctx = rfkt::cuda::init();

	auto dev = ctx.device();
	SPDLOG_INFO("Using device {}, CUDA {}.{}", dev.name(), dev.compute_major(), dev.compute_minor());
	SPDLOG_INFO("{} threads per MP, {} MPs, {} total threads", dev.max_threads_per_mp(), dev.mp_count(), dev.max_concurrent_threads());
	SPDLOG_INFO("{} bytes shared memory per block", dev.max_shared_per_block());
	SPDLOG_INFO("{} MHz", dev.clock_rate() / 1000);

	auto handler = [](int sig) {
		SPDLOG_INFO("Signal: {}", sig);
		break_loop = true;
	};

	signal(SIGINT, handler);

	// 53476

	auto km = rfkt::kernel_manager{};
	auto fc = rfkt::flame_compiler{ km };

	auto [tm_result, tm] = km.compile_file("assets/kernels/tonemap.cu",
		rfkt::compile_opts("tonemap")
		.function("tonemap")
		.flag(rfkt::compile_flag::extra_vectorization)
		.flag(rfkt::compile_flag::use_fast_math)
	);

	if (!tm_result.success) {
		SPDLOG_ERROR("{}", tm_result.log);
		return 1;
	}

	auto render_w = std::uint32_t{ 1280 * 8};
	auto render_h = std::uint32_t{ 720 * 8};

	//auto sesh = rfkt::nvenc::session::make();

	auto out_buf = std::make_shared<rfkt::cuda_buffer<uchar4>>(render_w * render_h);//sesh->initialize({ render_w, render_h }, { 1,30 });
	auto host_buf = std::vector<uchar4>(render_w * render_h);

	auto files = rfkt::fs::list("assets/flames_test/", rfkt::fs::filter::has_extension(".flam3"));

	int count = 0;
	for (const auto& filename : files)
	{
		if (break_loop) break;

		auto flame = rfkt::flame::import_flam3(filename.string());
		auto k_result = fc.get_flame_kernel(rfkt::precision::f32, &flame.value());

		if (!k_result.kernel.has_value()) {
			SPDLOG_ERROR("Could not compile kernel:\n{}\n-------------\n{}\n", k_result.log, k_result.source);
			return 1;
		}

		//SPDLOG_INFO("{}\n{}", k_result.source, k_result.log);
		auto& kernel = k_result.kernel.value();
		count++;

		auto state = kernel.warmup(rfkt::cuda::thread_local_stream(), flame.value(), {render_w, render_h}, 0.0, 1, 1.0/(30 * 8), 0xdeadbeef, 100);

		auto target_quality = 2048.0f;
		auto current_quality = 0.0f;
		std::size_t total_draws = 0, total_passes = 0;
		float elapsed_ms = 0;


		int seconds = 0;
		while (current_quality < target_quality && seconds < 120)
		{
			auto result = kernel.bin(rfkt::cuda::thread_local_stream(), state, target_quality - current_quality, 5000, 1'000'000'000);
			current_quality += result.quality;
			total_draws += result.total_draws;
			total_passes += result.total_passes;
			elapsed_ms += result.elapsed_ms;
			seconds+=5;
			SPDLOG_INFO("quality: {}", (int)current_quality);
		}

		CUDA_SAFE_CALL(tm.kernel().launch({ render_w / 8 + 1, render_h / 8 + 1, 1 }, { 8, 8, 1 }, rfkt::cuda::thread_local_stream())(
			state.bins.ptr(),
			out_buf->ptr(),
			render_w, render_h,
			static_cast<float>(flame->gamma.sample(0)),
			1.0f/(current_quality * sqrtf(10.0f)),
			static_cast<float>(flame->brightness.sample(0)),
			static_cast<float>(flame->vibrancy.sample(0))
			));
		out_buf->to_host_async(host_buf.data(), rfkt::cuda::thread_local_stream());
		cuStreamSynchronize(rfkt::cuda::thread_local_stream());

		std::string path = fmt::format("{}.png", filename.string());
		//stbi_write_bmp("test.bmp", render_w, render_h, 4, host_buf.data());
		rfkt::stbi::write_file(host_buf.data(), render_w, render_h, path);
		SPDLOG_INFO("{}: {} quality, {:.4} ms, {:.4}m iter/ms, {:.4}m draw/ms, {:.5}% eff\n", filename.filename().string(), current_quality, elapsed_ms, total_passes / elapsed_ms / 1'000'000, total_draws / elapsed_ms / 1'000'000, total_draws / float(total_passes) * 100.0f);
	}
}