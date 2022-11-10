#include <iostream>
#include <fmt/format.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util.h>
#include <librefrakt/util/stb.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/platform.h>
#include <signal.h>
#include <fstream>

#include <sstream>

#include <eznve.hpp>

#include <sol/sol.hpp>
#include <librefrakt/util/nvjpeg.h>

bool break_loop = false;

int mux(const rfkt::fs::path& in, const rfkt::fs::path& out, double fps) {

	auto args = fmt::format("./bin/mp4mux.exe --track \"{}\"#frame_rate={} {}", in.string(), fps, out.string());
	SPDLOG_INFO("Invoking muxer: {}", args);
	auto p = rfkt::platform::process{ {}, args };
	auto ret = p.wait_for_exit();
	SPDLOG_INFO("Muxer returned {}", ret);
	return ret;
}

namespace rfkt {
	class tonemapper {
	public:
		tonemapper(kernel_manager& km) {
			auto [tm_result, mod] = km.compile_file("assets/kernels/tonemap.cu",
				rfkt::compile_opts("tonemap")
				.function("tonemap<false>")
				.function("tonemap<true>")
				.flag(rfkt::compile_flag::extra_vectorization)
				.flag(rfkt::compile_flag::use_fast_math)
			);

			if (!tm_result.success) {
				SPDLOG_ERROR("{}", tm_result.log);
				exit(1);
			}

			tm = std::move(mod);
		}

		void run(CUdeviceptr bins, CUdeviceptr out, uint2 dims, double quality, bool planar, double gamma, double brightness, double vibrancy, CUstream stream) const {

			auto kernel = tm.kernel(planar ? "tonemap<true>" : "tonemap<false>");

			CUDA_SAFE_CALL(kernel.launch({ dims.x / 8 + 1, dims.y / 8 + 1, 1 }, { 8, 8, 1 }, stream)(
				bins,
				out,
				dims.x, dims.y,
				static_cast<float>(gamma),
				std::powf(10.0f, -log10f(quality) - 0.5f),
				static_cast<float>(brightness),
				static_cast<float>(vibrancy)
				));

		}
	private:
		cuda_module tm;
	};
}

rfkt::flame interpolate_dumb(const rfkt::flame& fl, const rfkt::flame& fr) {

	if (fl.xforms.size() > fr.xforms.size()) {
		auto diff = fl.xforms.size() - fr.xforms.size();
		for( int i = 0; i < diff; i++ ) 
	}

}

int main() {
	rfkt::flame_info::initialize("config/variations.yml");

	sol::state lua;
	lua.open_libraries(sol::lib::base);

	using namespace rfkt;

	auto ctx = rfkt::cuda::init();
	auto km = rfkt::kernel_manager{};
	auto fc = rfkt::flame_compiler{ km };
	auto jpeg = rfkt::nvjpeg::encoder{ rfkt::cuda::thread_local_stream() };
	auto tm = rfkt::tonemapper{ km };

	lua["render_jpeg"] = [&fc, &tm, &jpeg](sol::table args) -> bool {
		auto tls = rfkt::cuda::thread_local_stream();

		if (!args["flame"].valid()) return false;

		const flame& f = args["flame"];
		std::string output_file = args["out"];
		unsigned int width = args.get_or("width", 1920u);
		unsigned int height = args.get_or("height", 1080u);
		double quality = args.get_or("quality", 128.0);
		double time = args.get_or("time", 0.0);
		unsigned int jpeg_quality = args.get_or("jpeg_quality", 100u);
		unsigned int fps = args.get_or("fps", 30u);
		unsigned int loop_speed = args.get_or("loop_speed", 5000);
		unsigned int bin_time = args.get_or("bin_time", 2000);

		if (width > 3840) width = 3840;
		if (height > 2160) height = 2160;
		if (quality > 2000) quality = 2000;
		if (bin_time > 2000) bin_time = 2000;
		if (jpeg_quality <= 0) jpeg_quality = 100;
		if (jpeg_quality > 100) jpeg_quality = 100;

		auto k_result = fc.get_flame_kernel(rfkt::precision::f32, f);
		if (!k_result.kernel.has_value()) {
			SPDLOG_INFO("{}", k_result.source);
			SPDLOG_INFO("{}", k_result.log);
			return false;
		}

		auto kernel = rfkt::flame_kernel{ std::move(k_result.kernel.value()) };

		auto render = rfkt::cuda::make_buffer_async<uchar4>(width * height, tls);
		auto state = kernel.warmup(tls, f, { width, height }, time, 1, double(fps) / loop_speed, 0xdeadbeef, 100);

		auto result = kernel.bin(tls, state, quality, bin_time, 1'000'000'000).get();

		tm.run(state.bins.ptr(), render.ptr(), { width, height }, quality, true, f.gamma.sample(time), f.brightness.sample(time), f.vibrancy.sample(time), tls);
		auto data = jpeg.encode_image(render.ptr(), width, height, jpeg_quality, tls).get();

		fs::write(output_file, (const char *) data.data(), data.size(), false);
		return true;
	};


	auto ad_ut = lua.new_usertype<animated_double>("animated_double", sol::constructors<animated_double()>());
	ad_ut["t0"] = &animated_double::t0;

	auto aff_ut = lua.new_usertype<affine_matrix>("affine_matrix", sol::constructors<affine_matrix()>());
	aff_ut["a"] = &affine_matrix::a;
	aff_ut["d"] = &affine_matrix::d;
	aff_ut["b"] = &affine_matrix::b;
	aff_ut["e"] = &affine_matrix::e;
	aff_ut["c"] = &affine_matrix::c;
	aff_ut["f"] = &affine_matrix::f;
	aff_ut["identity"] = &affine_matrix::identity;
	aff_ut["rotate"] = &affine_matrix::rotate;
	aff_ut["scale"] = &affine_matrix::scale;
	aff_ut["translate"] = &affine_matrix::translate;

	auto vl_ut = lua.new_usertype<vlink>("vlink", sol::constructors<vlink()>());
	vl_ut["identity"] = &vlink::identity;
	vl_ut["aff_mod_rotate"] = &vlink::aff_mod_rotate;
	vl_ut["aff_mod_scale"] = &vlink::aff_mod_scale;
	vl_ut["variations"] = &vlink::variations;
	vl_ut["parameters"] = &vlink::parameters;

	auto xf_ut = lua.new_usertype<xform>("xform", sol::constructors<xform()>());
	xf_ut["identity"] = &xform::identity;
	xf_ut["weight"] = &xform::weight;
	xf_ut["color"] = &xform::color;
	xf_ut["color_speed"] = &xform::color_speed;
	xf_ut["opacity"] = &xform::opacity;
	xf_ut["vchain"] = &xform::vchain;

	auto f_ut = lua.new_usertype<flame>("flame", sol::constructors<flame()>());
	f_ut["xforms"] = &flame::xforms;
	f_ut["final_xform"] = &flame::final_xform;
	f_ut["scale"] = &flame::scale;
	f_ut["rotate"] = &flame::rotate;
	f_ut["gamma"] = &flame::gamma;
	f_ut["vibrancy"] = &flame::vibrancy;
	f_ut["brightness"] = &flame::brightness;
	f_ut["import_flam3"] = &flame::import_flam3;

	signal(SIGINT, [](int sig) {});

	auto exec = [&](std::string_view code) {
		return lua.script(code, [](lua_State*, sol::protected_function_result pfr) {
			return pfr;
			});
	};

	std::string input;
	while (true) {
		std::cout << "> ";
		std::getline(std::cin, input);
		if (input == "exit()") break;

		auto result = exec(input);
		std::cout << result.operator std::string() << std::endl;
	}
}

int mainy() {

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
		.function("tonemap<false>")
		.flag(rfkt::compile_flag::extra_vectorization)
		.flag(rfkt::compile_flag::use_fast_math)
	);

	if (!tm_result.success) {
		SPDLOG_ERROR("{}", tm_result.log);
		return 1;
	}

	auto render_w = std::uint32_t{ 1280 };
	auto render_h = std::uint32_t{ 720 };

	const int fps = 30;
	const int seconds = 1;

	//auto sesh = rfkt::nvenc::session::make();

	//auto out_buf = std::make_shared<rfkt::cuda_buffer<uchar4>>(render_w * render_h);//sesh->initialize({ render_w, render_h }, { 1,30 });
	auto files = rfkt::fs::list("assets/flames_test/", rfkt::fs::filter::has_extension(".flam3"));

	auto sesh = eznve::encoder(uint2{ render_w, render_h }, uint2{ fps, 1 }, eznve::codec::h264, (CUcontext)ctx, [](const eznve::encode_info&) {});

	int count = 0;
	for (const auto& filename : files)
	{
		if (break_loop) break;

		auto flame = rfkt::flame::import_flam3(filename.string());
		auto k_result = fc.get_flame_kernel(rfkt::precision::f32, flame.value());

		auto fprint = [&f = flame.value()](std::string_view str) {
			auto v = f.seek(str);
			if (v) SPDLOG_INFO("{}: {}", str, v->t0);
			else SPDLOG_INFO("{}: null", str);
		};

		if (!k_result.kernel.has_value()) {
			SPDLOG_ERROR("Could not compile kernel:\n{}\n-------------\n{}\n", k_result.log, k_result.source);
			return 1;
		}

		//fprint("f/0/v/spherical");
		//break;

		//SPDLOG_INFO("{}\n{}", k_result.source, k_result.log);
		auto& kernel = k_result.kernel.value();
		count++;

		int total_frames = fps * seconds;
		std::string path = fmt::format("{}.h264", filename.string());

		auto out_file = std::ofstream{};
		out_file.open(path, std::ios::out | std::ios::binary);

		sesh.set_callback([&out_file](const eznve::encode_info& info) {
			out_file.write((const char*)info.data.data(), info.data.size());
		});

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
				sesh.buffer(),
				render_w, render_h,
				static_cast<float>(flame->gamma.sample(0)),
				1.0f / (current_quality * sqrtf(10.0f)),
				static_cast<float>(flame->brightness.sample(0)),
				static_cast<float>(flame->vibrancy.sample(0))
				));
			cuStreamSynchronize(rfkt::cuda::thread_local_stream());
			auto frame_end = std::chrono::high_resolution_clock::now();
			auto total_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(frame_end - frame_start).count() / 1'000'000.0;

			sesh.submit_frame((frame % fps == 0)? eznve::frame_flag::idr : eznve::frame_flag::none);
			//stbi_write_bmp("test.bmp", render_w, render_h, 4, host_buf.data());
			//rfkt::stbi::write_file(host_buf.data(), render_w, render_h, path);
			SPDLOG_INFO("[{}%] {}: {} quality, {:.4} ms bin, {:.4} ms total, {:.4}m iter/ms, {:.4}m draw/ms, {:.5}% eff, {:.3} quality at 30fps", static_cast<int>(frame / float(total_frames) * 100.0f), filename.filename().string(), (int)current_quality, elapsed_ms, total_ms, total_passes / elapsed_ms / 1'000'000, total_draws / elapsed_ms / 1'000'000, total_draws / float(total_passes) * 100.0f, current_quality / elapsed_ms * 1000.0 / 30.0);
		}

		sesh.flush();
		out_file.close();
		auto new_path = filename.string() + ".mp4";
		auto muxret = mux(path, new_path, fps);
		//if(muxret == 0) std::filesystem::remove(path);

	}
}