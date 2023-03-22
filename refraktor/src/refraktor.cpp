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

#include <librefrakt/util/gpuinfo.h>
#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/tonemapper.h>
#include <librefrakt/image/converter.h>

bool break_loop = false;

int mux(const rfkt::fs::path& in, const rfkt::fs::path& out, double fps) {

	auto args = fmt::format("./bin/mp4mux.exe --track \"{}\"#frame_rate={} {}", in.string(), fps, out.string());
	SPDLOG_INFO("Invoking muxer: {}", args);
	auto p = rfkt::platform::process{ {}, args };
	auto ret = p.wait_for_exit();
	SPDLOG_INFO("Muxer returned {}", ret);
	return ret;
}

rfkt::flame interpolate_dumb(const rfkt::flame& fl, const rfkt::flame& fr) {

	auto ret = rfkt::flame{};

	/*ret.center.first = fl.center.first.make_interpolator(fr.center.first);
	ret.center.second = fl.center.second.make_interpolator(fr.center.second);
	ret.scale = fl.scale.make_interpolator(fr.scale);
	ret.rotate = fl.rotate.make_interpolator(fr.rotate);

	ret.gamma = fl.gamma.make_interpolator(fr.gamma);
	ret.brightness = fl.brightness.make_interpolator(fr.brightness);
	ret.vibrancy = fl.vibrancy.make_interpolator(fr.vibrancy);

	static auto interp_vl = [](const rfkt::vlink& vl, const rfkt::vlink& vr) {
		auto ret = rfkt::vlink{};
		ret.affine.a = vl.affine.a.make_interpolator(vr.affine.a);
		ret.affine.d = vl.affine.d.make_interpolator(vr.affine.d);
		ret.affine.b = vl.affine.b.make_interpolator(vr.affine.b);
		ret.affine.e = vl.affine.e.make_interpolator(vr.affine.e);
		ret.affine.c = vl.affine.c.make_interpolator(vr.affine.c);
		ret.affine.f = vl.affine.f.make_interpolator(vr.affine.f);
		
		ret.aff_mod_translate.first = vl.aff_mod_translate.first.make_interpolator(vr.aff_mod_translate.first);
		ret.aff_mod_translate.second = vl.aff_mod_translate.second.make_interpolator(vr.aff_mod_translate.second);
		ret.aff_mod_scale = vl.aff_mod_scale.make_interpolator(vr.aff_mod_scale);
		ret.aff_mod_rotate = vl.aff_mod_rotate.make_interpolator(vr.aff_mod_rotate);

		std::set<std::size_t> all_parameters;
		std::set<std::size_t> all_variations;

		for (auto& [id, _] : vl.variations) all_variations.insert(id);
		for (auto& [id, _] : vr.variations) all_variations.insert(id);
		for (auto& [id, _] : vl.parameters) all_parameters.insert(id);
		for (auto& [id, _] : vr.parameters) all_parameters.insert(id);

		for (auto id : all_variations) {
			const auto& left = (vl.variations.contains(id)) ? vl.variations.at(id) : rfkt::animated_double{};
			const auto& right = (vr.variations.contains(id)) ? vr.variations.at(id) : rfkt::animated_double{};

			ret.variations[id] = left.make_interpolator(right);
		}

		for (auto id : all_parameters) {
			const auto& left = (vl.parameters.contains(id)) ? vl.parameters.at(id) : rfkt::animated_double{};
			const auto& right = (vr.parameters.contains(id)) ? vr.parameters.at(id) : rfkt::animated_double{};

			ret.parameters[id] = left.make_interpolator(right);
		}

		return ret;
	};

	static auto interp_xf = [](const rfkt::xform& xl, const rfkt::xform& xr) {

		auto ret = rfkt::xform{};
		ret.weight = xl.weight.make_interpolator(xr.weight);
		ret.color = xl.color.make_interpolator(xr.color);
		ret.color_speed = xl.color_speed.make_interpolator(xr.color_speed);
		ret.opacity = xl.opacity.make_interpolator(xr.opacity);

		auto total_vls = std::max(xl.vchain.size(), xr.vchain.size());
		ret.vchain.reserve(total_vls);

		for (int i = 0; i < total_vls; i++) {
			const auto& left = (i < xl.vchain.size()) ? xl.vchain[i] : rfkt::vlink::identity();
			const auto& right = (i < xr.vchain.size()) ? xr.vchain[i] : rfkt::vlink::identity();

			auto xi = interp_vl(left, right);
			ret.vchain.emplace_back(std::move(xi));
		}

		return ret;
	};

	auto total_xf = std::max(fl.xforms.size(), fr.xforms.size());
	ret.xforms.reserve(total_xf);
	for (int i = 0; i < total_xf; i++) {
		const auto& left = (i < fl.xforms.size()) ? fl.xforms[i] : rfkt::xform::identity();
		const auto& right = (i < fr.xforms.size()) ? fr.xforms[i] : rfkt::xform::identity();

		ret.xforms.emplace_back(interp_xf(left, right));
	}

	if (fl.final_xform || fr.final_xform) {
		const auto& left = (fl.final_xform) ? fl.final_xform.value() : rfkt::xform::identity();
		const auto& right = (fr.final_xform) ? fr.final_xform.value() : rfkt::xform::identity();

		ret.final_xform = interp_xf(left, right);
	}

	auto& ret_pal = ret.palette();
	const auto& l_pal = fl.palette();
	const auto& r_pal = fr.palette();

	for (int i = 0; i < ret.palette().size(); i++) {
		ret_pal[i][0] = l_pal[i][0].make_interpolator(r_pal[i][0]);
		ret_pal[i][1] = l_pal[i][1].make_interpolator(r_pal[i][1]);
		ret_pal[i][2] = l_pal[i][2].make_interpolator(r_pal[i][2]);
	}*/

	return ret;
}

class pipeline {
public:



private:

	//rfkt::cuda_buffer

	rfkt::flame_compiler& fc;
	rfkt::tonemapper& tm;
	rfkt::denoiser& dn;
	ezrtc::cuda_module converter;
};

int main() {
	auto ctx = rfkt::cuda::init();

	rfkt::gpuinfo::init();

	auto dev = rfkt::gpuinfo::device::by_index(0);

	SPDLOG_INFO("clock: {}", dev.clock());
	SPDLOG_INFO("max clock: {}", dev.max_clock());

	rfkt::flame_info::initialize("config");

	sol::state lua;
	lua.open_libraries(sol::lib::base, sol::lib::math, sol::lib::string);

	using namespace rfkt;

	rfkt::denoiser::init(ctx);
	auto dn = rfkt::denoiser{ { 1920, 1080 }, false };

	auto km = ezrtc::compiler{std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::sqlite_cache>("k2.sqlite3"))};
	km.find_system_cuda();
	km.add_include_path("assets/kernels/include");

	auto jpeg_stream = rfkt::cuda_stream{};

	auto fc = rfkt::flame_compiler{ km };
	auto jpeg = rfkt::nvjpeg::encoder{ jpeg_stream };
	auto tm = rfkt::tonemapper{ km };
	auto conv = rfkt::converter{ km };

	for (const auto& f : rfkt::fs::list("assets/flames", rfkt::fs::filter::has_extension(".flam3"))) {
		auto fopt = rfkt::flame::import_flam3(f.string());
		if (!fopt) {
			continue;
		}

		auto result = fc.get_flame_kernel(rfkt::precision::f32, fopt.value());
		if (!result.kernel) {
			SPDLOG_ERROR("\n{}", result.source);
			SPDLOG_ERROR("\n{}", result.log);
			return 1;
		}
	}


	lua["use_logging"] = [](bool log) {
		spdlog::set_level((log) ? spdlog::level::info: spdlog::level::off);
	};

	lua["render_jpeg"] = [&fc, &tm, &jpeg, &dn, &conv, &tls = jpeg_stream](sol::table args) -> bool {

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
		bool denoise = args.get_or("denoise", true);

		if (width > 3840) width = 3840;
		if (height > 2160) height = 2160;
		if (quality > 200000) quality = 200000;
		if (bin_time > 60000) bin_time = 60000;
		if (jpeg_quality <= 0) jpeg_quality = 1;
		if (jpeg_quality > 100) jpeg_quality = 100;

		auto k_result = fc.get_flame_kernel(rfkt::precision::f32, f);

		if (!k_result.kernel.has_value()) {
			SPDLOG_INFO("\n{}", k_result.source);
			SPDLOG_INFO("{}", k_result.log);

			return false;
		}

		auto kernel = rfkt::flame_kernel{ std::move(k_result.kernel.value()) };


		auto render = rfkt::cuda_buffer<half3>{ width * height, tls };
		rfkt::timer t;
		auto state = kernel.warmup(tls, f, { width, height }, time, 1, double(fps) / loop_speed, 0xdeadbeef, 100);
		cuStreamSynchronize(tls);
		auto warmup_time = t.count();
		t.reset();

		auto result = kernel.bin(tls, state, quality, bin_time, 1'000'000'000).get();
		auto rbin_time = t.count();

		double mdraws_per_ms = result.total_draws / result.elapsed_ms / 1'000'000;
		SPDLOG_INFO("mdraws per ms: {}", mdraws_per_ms);

		t.reset();
		tm.run(state.bins, render, {width, height}, result.quality, f.gamma.sample(time), f.brightness.sample(time), f.vibrancy.sample(time), tls);
		cuStreamSynchronize(tls);
		auto tm_time = t.count();

		auto out = rfkt::cuda_buffer<uchar4>(width * height, tls);
		auto smooth = rfkt::cuda_buffer<half3>(width * height, tls);

		auto dn_time = (denoise) ? dn.denoise({ width, height }, render, smooth, tls).get() : 0.0;

		t.reset();
		conv.to_24bit((denoise) ? smooth : render, out, { width, height }, true, tls);
		cuStreamSynchronize(tls);
		auto conv_time = t.count();

		SPDLOG_INFO("warmup: {} bin: {} tonemap: {} denoise: {} convert: {}", warmup_time, rbin_time, tm_time, dn_time, conv_time);

		auto data = jpeg.encode_image(out.ptr(), width, height, jpeg_quality, tls).get();

		fs::write(output_file, (const char *) data.data(), data.size(), false);
		return true;
	};

	lua["interpolate_dumb"] = &interpolate_dumb;

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
	//vl_ut["variations"] = &vlink::variations;
	//vl_ut["parameters"] = &vlink::parameters;

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
	f_ut["ez_import"] = [](unsigned int id) {
		return flame::import_flam3(fmt::format("assets/flames/electricsheep.247.{}.flam3", id));
	};

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

		auto result = exec(fmt::format("print((function() {} end)())", input));
		if(result.valid() && result.return_count() > 0)
			std::cout << result.get<std::string>(0) << std::endl;
	}
}

int mainy() {

	/*rfkt::flame_info::initialize("config/variations.yml");
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

	auto sesh = eznve::encoder(uint2{ render_w, render_h }, uint2{ fps, 1 }, eznve::codec::h264, (CUcontext)ctx, [](const eznve:chunk_info&) {});

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

		sesh.set_callback([&out_file](const eznve::chunk_info& info) {
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

	}*/

return 0;
}