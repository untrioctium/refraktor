#include <iostream>
#include <fmt/format.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util.h>
#include <librefrakt/util/stb.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/nvenc.h>

#include <signal.h>

struct iterator {
	float2 position;
	float color;
};


template<std::size_t threads_per_block>
struct thread_states_t {
	iterator iterators[threads_per_block];
	uint4 rand_states[threads_per_block];
	std::uint16_t shuffle_vote[threads_per_block];
	std::uint16_t shuffle[threads_per_block];
	std::uint8_t xform_vote[threads_per_block];
};

template <std::size_t threads_per_block, std::size_t flame_size_reals>
struct shared_state_t {
	thread_states_t<threads_per_block> ts;

	float flame[flame_size_reals];
	uchar3 palette[256];

	float2 antialiasing_offsets;
	unsigned long long tss_quality;
	unsigned long long tss_passes;
	unsigned long long tss_start;
	bool should_bail;
};

constexpr auto iterator_size = sizeof(iterator);
constexpr auto thread_states_size = sizeof(thread_states_t<96>);
constexpr auto shared_state_size = sizeof(shared_state_t<96, 56>);

bool break_loop = false;

int main() {

	rfkt::flame_info::initialize("config/variations.yml");
	auto ctx = rfkt::cuda::init();


	SPDLOG_INFO("Running on {}, {} threads per MP, {} MPs", ctx.device().name(), ctx.device().max_threads_per_mp(), ctx.device().mp_count());
	SPDLOG_INFO("Cuda version {}.{}", ctx.device().compute_major(), ctx.device().compute_minor());

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

	auto render_w = std::uint32_t{ 1280};
	auto render_h = std::uint32_t{ 720};

	//auto sesh = rfkt::nvenc::session::make();

	auto out_buf = std::make_shared<rfkt::cuda_buffer<uchar4>>(render_w * render_h);//sesh->initialize({ render_w, render_h }, { 1,30 });
	auto host_buf = std::vector<uchar4>(render_w * render_h);

	auto files = rfkt::fs::list("assets/flames/", rfkt::fs::filter::has_extension(".flam3"));

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

		SPDLOG_INFO("{}\n{}", k_result.source, k_result.log);
		auto& kernel = k_result.kernel.value();
		count++;

		auto state = kernel.warmup(rfkt::cuda::thread_local_stream(), flame.value(), {render_w, render_h}, 0.0, 1, 1.0/(30 * 8), 0xdeadbeef, 100);

		auto target_quality = 128.0f;
		auto current_quality = 0.0f;
		std::size_t total_draws = 0, total_passes = 0;
		float elapsed_ms = 0;


		int seconds = 0;
		while (current_quality < target_quality && seconds < 5)
		{
			auto result = kernel.bin(rfkt::cuda::thread_local_stream(), state, target_quality - current_quality, 1000, 1'000'000'000);
			current_quality += result.quality;
			total_draws += result.total_draws;
			total_passes += result.total_passes;
			elapsed_ms += result.elapsed_ms;
			seconds++;
			SPDLOG_INFO("quality: {}", (int)current_quality);
		}

		tm.kernel().launch({ render_w / 8 + 1, render_h / 8 + 1, 1 }, { 8, 8, 1 }, rfkt::cuda::thread_local_stream())(
			state.bins.ptr(),
			out_buf->ptr(),
			render_w, render_h,
			static_cast<float>(flame->gamma.sample(0)),
			std::powf(10.0f, -log10f(current_quality) - 0.5f),
			static_cast<float>(flame->brightness.sample(0)),
			static_cast<float>(flame->vibrancy.sample(0))
			);
		out_buf->to_host_async(host_buf.data(), rfkt::cuda::thread_local_stream());
		cuStreamSynchronize(rfkt::cuda::thread_local_stream());

		std::string path = fmt::format("{}.png", filename.string());
		//stbi_write_bmp("test.bmp", render_w, render_h, 4, host_buf.data());
		rfkt::stbi::write_file(host_buf.data(), render_w, render_h, path);
		SPDLOG_INFO("{} quality, {} ms, {}m iter/ms, {} draw/ms\n", current_quality, elapsed_ms, total_passes / elapsed_ms / 1'000'000, total_draws / elapsed_ms / 1'000'000);
		SPDLOG_INFO("{:.4}%", float(count) / files.size() * 100.0f);
	}
}