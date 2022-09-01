#include <iostream>
#include <fmt/format.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util.h>
#include <librefrakt/util/stb.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/platform.h>
#include <signal.h>
#include <fstream>

#include <librefrakt/util/nvenc.h>

bool break_loop = false;

int mux(const rfkt::fs::path& in, const rfkt::fs::path& out, double fps) {

	auto args = fmt::format("./bin/mp4mux.exe --track \"{}\"#frame_rate={} {}", in.string(), fps, out.string());
	SPDLOG_INFO("Invoking muxer: {}", args);
	auto p = rfkt::platform::process{ {}, args };
	auto ret = p.wait_for_exit();
	SPDLOG_INFO("Muxer returned {}", ret);
	return ret;
}

constexpr auto split_path(std::string_view sv) -> std::pair<std::string_view, std::string_view> {
	
	auto pos = sv.find('/');
	if (pos == std::string_view::npos) return { sv, {} };

	auto head = sv.substr(0, pos);
	if (pos + 1 >= sv.length()) return { head, {} };

	return { head, sv.substr(pos + 1) };
}

constexpr int to_int(std::string_view sv) {
	int ret = 0;
	for (const auto ch : sv) {
		if (ch < '0' || ch > '9') return -1;
		ret *= 10;
		ret += ch - '0';
	}

	return ret;
}

rfkt::animated_double* seek(rfkt::flame& f, std::string_view path) {
	auto [head, tail] = split_path(path);
	
	switch (head[0]) {
	case 'x': return (head.size() == 1 && tail.empty()) ? &f.center.first : nullptr;
	case 'y': return (head.size() == 1 && tail.empty()) ? &f.center.second : nullptr;
	case 's': return (head.size() == 1 && tail.empty()) ? &f.scale : nullptr;
	case 'r': return (head.size() == 1 && tail.empty()) ? &f.rotate : nullptr;
	case 'g': return (head.size() == 1 && tail.empty()) ? &f.gamma: nullptr;
	case 'v': return (head.size() == 1 && tail.empty()) ? &f.vibrancy : nullptr;
	case 'b': return (head.size() == 1 && tail.empty()) ? &f.brightness : nullptr;
	}
}

rfkt::animated_double* seek(rfkt::vlink& vl, std::string_view path) {

	auto [head, tail] = split_path(path);

	if (tail.empty() || head.size() > 1) return nullptr;

	switch (head[0]) {
	case 'v': {
		if (!rfkt::flame_info::is_variation(tail)) return nullptr;

		auto iter = vl.variations.find(rfkt::flame_info::variation(tail).index);
		if (iter == vl.variations.end()) return nullptr;
		return &iter->second;
	}
	case 'p': {
		if (!rfkt::flame_info::is_parameter(tail)) return nullptr;

		auto iter = vl.parameters.find(rfkt::flame_info::parameter(tail).index);
		if (iter == vl.parameters.end()) return nullptr;
		return &iter->second;
	}
	case 'a':
		if (tail.size() > 1) return nullptr;
		switch (tail[0]) {
		case 'a': return &vl.affine.a;
		case 'b': return &vl.affine.b;
		case 'c': return &vl.affine.c;
		case 'd': return &vl.affine.d;
		case 'e': return &vl.affine.e;
		case 'f': return &vl.affine.f;
		default: return nullptr;
		}

	case 'm':
		if (tail.size() > 1) return nullptr;
		switch (tail[0]) {
		case 'r': return &vl.aff_mod_rotate;
		case 's': return &vl.aff_mod_scale;
		case 'x': return &vl.aff_mod_translate.first;
		case 'y': return &vl.aff_mod_translate.second;
		default: return nullptr;
		}

	default: return nullptr;
	}
}

rfkt::animated_double* seek(rfkt::xform& xf, std::string_view path) {
	auto [head, tail] = split_path(path);

	switch (head[0]) {
	case 'w': return (head.size() == 1 && tail.empty())? &xf.weight: nullptr;
	case 'c':
		if (head.size() == 1) return &xf.color;
		if (head.size() != 2) return nullptr;

		if (head[1] == 's') return &xf.color_speed;
		if (head[2] == 'h') {
			if (tail.empty()) return nullptr;

			auto [t_head, t_tail] = split_path(tail);
			if (t_tail.empty()) return nullptr;

			auto idx = to_int(t_head);
			if (idx < 0 || idx >= xf.vchain.size()) return nullptr;
			return seek(xf.vchain[idx], t_tail);
		}

		return nullptr;

	case 'o': return (head.size() == 1 && tail.empty())? &xf.opacity : nullptr;
	default: return nullptr;
	}
}



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

	auto render_w = std::uint32_t{ 1280 };
	auto render_h = std::uint32_t{ 720 };

	const int fps = 60;
	const int seconds = 5;

	//auto sesh = rfkt::nvenc::session::make();

	//auto out_buf = std::make_shared<rfkt::cuda_buffer<uchar4>>(render_w * render_h);//sesh->initialize({ render_w, render_h }, { 1,30 });
	auto files = rfkt::fs::list("assets/flames_test/", rfkt::fs::filter::has_extension(".flam3"));

	int count = 0;
	for (const auto& filename : files)
	{
		if (break_loop) break;

		auto flame = rfkt::flame::import_flam3(filename.string());
		auto k_result = fc.get_flame_kernel(rfkt::precision::f32, flame.value());

		if (!k_result.kernel.has_value()) {
			SPDLOG_ERROR("Could not compile kernel:\n{}\n-------------\n{}\n", k_result.log, k_result.source);
			return 1;
		}

		//SPDLOG_INFO("{}\n{}", k_result.source, k_result.log);
		auto& kernel = k_result.kernel.value();
		count++;

		int total_frames = fps * seconds;
		std::string path = fmt::format("{}.h264", filename.string());

		auto sesh = rfkt::nvenc::session::make();
		auto buf = sesh->initialize({ render_w, render_h }, { fps, 1 });

		auto out_file = std::ofstream{};
		out_file.open(path, std::ios::out | std::ios::binary);

		for (int frame = 0; frame < total_frames; frame++)
		{
			auto frame_start = std::chrono::high_resolution_clock::now();
			auto state = kernel.warmup(rfkt::cuda::thread_local_stream(), flame.value(), { render_w, render_h }, frame/float(total_frames), 1, 1.0 / (total_frames), 0xdeadbeef, 100);

			auto target_quality = 2048.0;
			auto current_quality = 0.0f;
			std::size_t total_draws = 0, total_passes = 0;
			float elapsed_ms = 0;


			int seconds = 0;
			while (current_quality < target_quality && seconds < 5)
			{
				auto result = kernel.bin(rfkt::cuda::thread_local_stream(), state, target_quality - current_quality, 25, 1'000'000'000).get();
				current_quality += result.quality;
				total_draws += result.total_draws;
				total_passes += result.total_passes;
				elapsed_ms += result.elapsed_ms;
				seconds += 5;
				//SPDLOG_INFO("quality: {}", (int)current_quality);
			}

			CUDA_SAFE_CALL(tm.kernel().launch({ render_w / 8 + 1, render_h / 8 + 1, 1 }, { 8, 8, 1 }, rfkt::cuda::thread_local_stream())(
				state.bins.ptr(),
				buf->ptr(),
				render_w, render_h,
				static_cast<float>(flame->gamma.sample(0)),
				1.0f / (current_quality * sqrtf(10.0f)),
				static_cast<float>(flame->brightness.sample(0)),
				static_cast<float>(flame->vibrancy.sample(0))
				));
			cuStreamSynchronize(rfkt::cuda::thread_local_stream());
			auto frame_end = std::chrono::high_resolution_clock::now();
			auto total_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(frame_end - frame_start).count() / 1'000'000.0;

			if (auto ret = sesh->submit_frame(frame % fps == 0); ret) {
				out_file.write((const char*)ret->data(), ret->size());
			}
			//stbi_write_bmp("test.bmp", render_w, render_h, 4, host_buf.data());
			//rfkt::stbi::write_file(host_buf.data(), render_w, render_h, path);
			SPDLOG_INFO("[{}%] {}: {} quality, {:.4} ms bin, {:.4} ms total, {:.4}m iter/ms, {:.4}m draw/ms, {:.5}% eff, {:.3} quality at 30fps", static_cast<int>(frame / float(total_frames) * 100.0f), filename.filename().string(), (int)current_quality, elapsed_ms, total_ms, total_passes / elapsed_ms / 1'000'000, total_draws / elapsed_ms / 1'000'000, total_draws / float(total_passes) * 100.0f, current_quality / elapsed_ms * 1000.0 / 30.0);
		}

		if (auto ret = sesh->submit_frame(false, true); ret) {
			out_file.write((const char*)ret->data(), ret->size());
		}
		out_file.close();
		auto new_path = filename.string() + ".mp4";
		auto muxret = mux(path, new_path, fps);
		//if(muxret == 0) std::filesystem::remove(path);

	}
}