#include <iostream>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util.h>
#include <librefrakt/util/stb.h>
#include <librefrakt/util/filesystem.h>
#include <signal.h>
#include <fstream>
#include <source_location>

#include <sstream>

#include <eznve.hpp>

#include <sol/sol.hpp>
#include <librefrakt/interface/jpeg_encoder.h>

#include <librefrakt/util/gpuinfo.h>
#include <librefrakt/interface/denoiser.h>
#include <librefrakt/image/tonemapper.h>
#include <librefrakt/image/converter.h>
#include <librefrakt/util/zlib.h>
#include <librefrakt/anima.h>

#include <flang/grammar.h>

#include <roccu.h>

bool break_loop = false;

/*int mux(const rfkt::fs::path& in, const rfkt::fs::path& out, double fps) {

	auto args = fmt::format("./bin/mp4mux.exe --track \"{}\"#frame_rate={} {}", in.string(), fps, out.string());
	SPDLOG_INFO("Invoking muxer: {}", args);
	auto p = rfkt::platform::process{ {}, args };
	auto ret = p.wait_for_exit();
	SPDLOG_INFO("Muxer returned {}", ret);
	return ret;
}*/

class interpolator {
public:
	interpolator(const rfkt::flame& linit, const rfkt::flame& rinit, const rfkt::flamedb& fdb, bool interp_by_weight)
	{
		left.flame = linit;
		right.flame = rinit;

		left.type_hash = linit.hash();
		right.type_hash = rinit.hash();

		left.value_hash = linit.value_hash();
		right.value_hash = rinit.value_hash();

		rebuild_sides(interp_by_weight, fdb);

		for (int i = 0; i < left.flame.palette.size(); i++) {
			auto diff = right.flame.palette[i][0] - left.flame.palette[i][0];

			if (diff > 180) {
				right.flame.palette[i][0] -= 360;
			}
			else if (diff < -180) {
				right.flame.palette[i][0] += 360;
			}
		}
	}

	const auto& left_flame() const { return left.flame; }
	const auto& right_flame() const { return right.flame; }

	template<typename Packer, typename Invoker>
	void pack_samples(Packer& p, Invoker& i, double start, double offset, int count, int w, int h, double mix) const {

		auto l_samples = std::vector<double>{};
		auto r_samples = std::vector<double>{};

		auto lpack = [&l_samples](auto s) {
			l_samples.push_back(s);
		};

		left.flame.pack_samples(lpack, i, start, offset, count, w, h);

		auto rpack = [&r_samples](auto s) {
			r_samples.push_back(s);
		};
		right.flame.pack_samples(rpack, i, start, offset, count, w, h);

		for (int i = 0; i < l_samples.size(); i++) {
			p(l_samples[i] * (1 - mix) + r_samples[i] * mix);
		}

	}

	template<typename Invoker>
	double interp_anima(const rfkt::accessor& at, Invoker& i, double t, double mix) const {
		const auto* left = at.access(left.flame);
		const auto* right = at.access(right.flame);

		if (!left || !right) {
			SPDLOG_ERROR("interp_anima: left or right flame field not found: {}", at.to_string());
			return 0.0;
		}

		return left->sample(t, i) * (1 - mix) + right->sample(t, i) * mix;
	}

	template<typename Invoker>
	double interp_anima(rfkt::anima_ptr<rfkt::flame> ptr, Invoker& i, double t, double mix) const {
		return (left.flame.*ptr).sample(t, i) * (1 - mix) + (right.flame.*ptr).sample(t, i) * mix;
	}

private:

	static void interp_xforms(rfkt::xform& l, rfkt::xform& r, const rfkt::flamedb& fdb) {

		const auto max_vlinks = std::max(l.vchain.size(), r.vchain.size());

		while (l.vchain.size() < max_vlinks) {
			l.vchain.emplace_back(fdb.make_padder(r.vchain[l.vchain.size()]));
		}

		while (r.vchain.size() < max_vlinks) {
			r.vchain.emplace_back(fdb.make_padder(l.vchain[r.vchain.size()]));
		}

		for (int i = 0; i < max_vlinks; i++) {

			auto& vll = l.vchain[i];
			auto& vlr = r.vchain[i];

			for (const auto& [name, vdata] : vll) {
				if (!vlr.has_variation(name)) {
					vlr.add_variation({ name, vdata });
					vlr[name].weight = 0.0;
				}
			}

			for (const auto& [name, vdata] : vlr) {
				if (!vll.has_variation(name)) {
					vll.add_variation({ name, vdata });
					vll[name].weight = 0.0;
				}
			}
		}
	}

	void rebuild_sides(bool interp_by_weight, const rfkt::flamedb& fdb) {
		
		const auto max_xforms = std::max(left.flame.xforms().size(), right.flame.xforms().size());

		auto nleft = left.flame.xforms().size();
		auto nright = right.flame.xforms().size();

		if (interp_by_weight) {
			for (int i = 0; i < right.flame.xforms().size(); i++) {
				auto xfc = right.flame.xforms()[i];
				left.flame.add_xform(std::move(xfc));
			}

			right.flame.clear_xforms();
			for (int i = 0; i < left.flame.xforms().size(); i++) {
				auto xfc = left.flame.xforms()[i];
				right.flame.add_xform(std::move(xfc));
			}

			for (int i = nleft; i < left.flame.xforms().size(); i++) {
				left.flame.xforms()[i].weight = 0.0;
			}

			for (int i = 0; i < nleft; i++) {
				right.flame.xforms()[i].weight = 0.0;
			}
		}
		else {
			while (left.flame.xforms().size() < max_xforms) {
				left.flame.add_xform({});
			}

			while (right.flame.xforms().size() < max_xforms) {
				right.flame.add_xform({});
			}

			for (int i = 0; i < max_xforms; i++) {
				interp_xforms(left.flame.xforms()[i], right.flame.xforms()[i], fdb);
			}
		}

		if (left.flame.final_xform.has_value() && !right.flame.final_xform.has_value()) {
			right.flame.final_xform = rfkt::xform{};
		}

		if (right.flame.final_xform.has_value() && !left.flame.final_xform.has_value()) {
			left.flame.final_xform = rfkt::xform{};
		}

		if (left.flame.final_xform.has_value() && right.flame.final_xform.has_value()) {
			interp_xforms(left.flame.final_xform.value(), right.flame.final_xform.value(), fdb);
		}
	}

	struct side {
		rfkt::flame flame;
		rfkt::hash_t value_hash;
		rfkt::hash_t type_hash;
	};

	side left;
	side right;

};

namespace rfkt {
	class postprocessor {
	public:
		postprocessor(ezrtc::compiler& kc, uint2 dims, roccu::gpu_stream& stream, rfkt::denoiser_flag::flags dn_opts = rfkt::denoiser_flag::none) :
			tm(kc),
			dn(rfkt::denoiser::make("rfkt::oidn_denoiser", uint2{dims.x, dims.y}, dn_opts, stream)),
			conv(kc),
			tonemapped(dn_opts& rfkt::denoiser_flag::upscale ? dims.x / 2 : dims.x, dn_opts& rfkt::denoiser_flag::upscale ? dims.y / 2 : dims.y),
			denoised(dims.x, dims.y),
			dims_(dims),
			upscale(dn_opts & rfkt::denoiser_flag::upscale)
		{

		}

		~postprocessor() = default;

		postprocessor(const postprocessor&) = delete;
		postprocessor& operator=(const postprocessor&) = delete;

		postprocessor(postprocessor&&) = default;
		postprocessor& operator=(postprocessor&&) = default;

		auto make_output_buffer() const -> roccu::gpu_buffer<uchar3> {
			return roccu::gpu_buffer<uchar3>{ dims_.x* dims_.y };
		}

		auto make_output_buffer(RUstream stream) const -> roccu::gpu_buffer<uchar3> {
			return { dims_.x * dims_.y, stream };
		}

		auto input_dims() const -> uint2 {
			if (upscale) {
				return { dims_.x / 2, dims_.y / 2 };
			}
			else {
				return dims_;
			}
		}

		auto output_dims() const -> uint2 {
			return dims_;
		}

		std::future<double> post_process(
			roccu::gpu_span<float4> in,
			roccu::gpu_span<uchar3> out,
			double quality,
			double gamma, double brightness, double vibrancy,
			bool planar_output,
			roccu::gpu_stream& stream) {

			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func(
				[&t = perf_timer]() mutable {
					t.reset();
				});

			tm.run(in, tonemapped, { quality, gamma, brightness, vibrancy }, stream);
			auto dnt = dn->denoise(tonemapped, denoised, dn_done).get();
			SPDLOG_INFO("denoise time: {}", dnt);
			conv.to_24bit(denoised, out, stream);
			stream.host_func([&t = perf_timer, p = std::move(promise)]() mutable {
				p.set_value(t.count());
				});

			return future;
		}

	private:
		rfkt::tonemapper tm;
		std::unique_ptr<rfkt::denoiser> dn;
		rfkt::converter conv;

		rfkt::gpu_image<half3> tonemapped;
		rfkt::gpu_image<half3> denoised;

		rfkt::timer perf_timer;

		roccu::gpu_event dn_done;

		uint2 dims_;
		bool upscale;
	};
}

int main() {

	auto ctx = rfkt::cuda::init();

	SPDLOG_INFO("Loaded on device: {}", ctx.device().name());

	//rfkt::denoiser::init(ctx);

	rfkt::flamedb fdb;
	rfkt::initialize(fdb, "config");

	auto stream = roccu::gpu_stream{};
	auto kernel = std::make_shared<ezrtc::sqlite_cache>((rfkt::fs::user_local_directory() / "kernel.sqlite3").string());
	auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(kernel);
	auto km = std::make_shared<ezrtc::compiler>(zlib);

	uint2 dims = { 3840, 2160 };

	auto pp = rfkt::postprocessor{ *km, dims, stream };
	//rfkt::cuda::check_and_download_cudart();
	//km->find_system_cuda();
	auto fc = rfkt::flame_compiler{ km };

	SPDLOG_INFO("notepad found: {}", rfkt::fs::command_in_path("notepad.exe") ? "yes" : "no");

	auto api = roccuGetApi();
	std::pair<std::size_t, std::string_view> best_encoder;
	best_encoder.first = std::numeric_limits<std::size_t>::max();

	for (auto name : rfkt::jpeg_encoder::names()) {
		auto meta = rfkt::jpeg_encoder::meta_for(name);
		if (meta->supported_apis.contains(api) && meta->priority < best_encoder.first) {
			best_encoder = { meta->priority, name };
		}
	}

	SPDLOG_INFO("Select jpeg encoder: {}", best_encoder.second);

	auto je = rfkt::jpeg_encoder::make(best_encoder.second, stream);

	SPDLOG_INFO("encoder: {}, prio: {}", je->name(), je->meta().priority);

	//roccuPrintAllocations();

	auto bins = rfkt::gpu_image<float4>{ pp.input_dims().x, pp.input_dims().y };
	auto out = rfkt::gpu_image<uchar3>{ dims.x, dims.y };

	constexpr static auto fps = 60;
	constexpr static auto seconds_per_loop = 5;
	constexpr static auto frames_per_loop = fps * seconds_per_loop;
	const auto loops_per_frame = 1.0 / frames_per_loop;
	const auto num_loops = 4;
	const auto num_frames = num_loops * seconds_per_loop * fps;

	auto functions = rfkt::function_table{};

	functions.add_or_update("increase", {
	{{"per_loop", {rfkt::func_info::arg_t::decimal, 360.0}}},
	"return iv + t * per_loop"
		});
	functions.add_or_update("sine", {
		{
			{"frequency", {rfkt::func_info::arg_t::decimal, 1.0}},
			{"amplitude", {rfkt::func_info::arg_t::decimal, 1.0}},
			{"phase", {rfkt::func_info::arg_t::decimal, 0.0}},
			{"sharpness", {rfkt::func_info::arg_t::decimal, 0.0}},
			{"absolute", {rfkt::func_info::arg_t::boolean, false}}
		},
		"local v = math.sin(t * frequency * math.pi * 2.0 + math.rad(phase))\n"
		"if sharpness > 0 then v = math.copysign(1.0, v) * (math.abs(v) ^ sharpness) end\n"
		"if absolute then v = math.abs(v) end\n"
		"return iv + v * amplitude\n"
		});

	auto invoker = functions.make_invoker();

	auto must_load_flame = [&](const rfkt::fs::path& path) -> rfkt::flame {
		auto fxml = rfkt::fs::read_string(path);
		auto f = rfkt::import_flam3(fdb, fxml);

		if (!f) {
			SPDLOG_ERROR("failed to import flame");
			exit(1);
		}

		return f.value();
	};

	//auto left = must_load_flame("assets/flames_test/electricsheep.247.19450.flam3");
	//auto right = must_load_flame("assets/flames_test/electricsheep.247.31464.flam3");

	auto left = must_load_flame("assets/flames_test/electricsheep.247.32660.flam3");
	auto right = must_load_flame("assets/flames_test/electricsheep.247.34338.flam3");


	auto interp = interpolator{ left, right, fdb, false };

	auto k = fc.get_flame_kernel(fdb, rfkt::precision::f32, interp.left_flame());

	constexpr auto smoothstep = [](double t, int steps) {

		constexpr static auto pascal = [](double a, int b) {
			double result = 1;
			for (int i = 0; i < b; i++) {
				result *= (a - i) / (i + 1);
			}
			return result;
		};

		double result = 0;

		for (int i = 0; i <= steps; i++) {
			result += pascal(-steps - 1, i) * pascal(2 * steps + 1, steps - i) * std::pow(t, steps + i + 1);
		}

		return result;
	};

	if (!k.kernel) {
		SPDLOG_INFO("\n{}\n{}", k.source, k.log);
		return -1;
	}

		//if (!k.kernel) { return -1; }

		//SPDLOG_INFO("left: {}\n", left.serialize().dump(2));

	//CUDA_SAFE_CALL(ruStreamSetAttribute(stream, RU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW, &stream_value));

	/*for (const auto& fname : rfkt::fs::list("assets/flames/", rfkt::fs::filter::has_extension(".flam3"))) {
		auto fxml = rfkt::fs::read_string(fname);
		auto f = rfkt::import_flam3(fdb, fxml);

		if (!f) {
			SPDLOG_ERROR("failed to import flame {}", fname.string());
			continue;
		}

		SPDLOG_INFO("compiling flame: {}", fname.string());
		auto k = fc.get_flame_kernel(fdb, rfkt::precision::f32, f.value());
	}*/

	for (int i = 0; i < num_frames; i++) {
		SPDLOG_INFO("frame {}", i);

		/*auto loop_id = i / frames_per_loop;
		double mix = 0;
		if (loop_id % 2 == 0) {
			mix = (loop_id / 2) % 2;
		}
		else {
			mix = (i % frames_per_loop) / double(frames_per_loop);

			if((loop_id - 1)/2 % 2 == 1) mix = 1 - mix;
		}*/

		double mix = i / double(num_frames);

		mix = smoothstep(mix, 10);

		auto t = i* loops_per_frame;
		auto samples = std::vector<double>{};
		auto packer = [&samples](double v) { samples.push_back(v); };
		double fudge = fps / 24.0;
		interp.pack_samples(packer, invoker, t - fudge * loops_per_frame, fudge* loops_per_frame, 4, pp.input_dims().x, pp.input_dims().y, mix);
		 
		auto state = k.kernel->warmup(stream, samples, std::move(bins), 0xDEADBEEF, 100);
		auto q = k.kernel->bin(stream, state, { .millis = 1000, .quality = 32 }).get();

		double gamma = interp.interp_anima(&rfkt::flame::gamma, invoker, t, mix);
		double brightness = interp.interp_anima(&rfkt::flame::brightness, invoker, t, mix);
		double vibrancy = interp.interp_anima(&rfkt::flame::vibrancy, invoker, t, mix);

		auto res = pp.post_process(state.bins, out, q.quality, gamma, brightness, vibrancy, true, stream).get();

		auto vec = je->encode_image(out, 100, stream).get()();

		//std::vector<uchar3> vec(out.area());
		//out.to_host(vec);


		auto basename = std::format("frame_{:04d}.jpg", i);
		rfkt::fs::write("testrender/" + basename, (const char*)vec.data(), vec.size(), false);

		//rfkt::stbi::write_file(vec.data(), pp.output_dims().x, pp.output_dims().y, "testrender/" + basename);

		bins = std::move(state.bins);
		bins.clear(stream);

		SPDLOG_INFO("{:.5f}, {:.5f} miters/ms {:.5f} mdraws/ms, {:.5f} quality/ms, {:.5f} ppt", q.quality, q.total_passes / (1'000'000 * q.elapsed_ms), q.total_draws / (1'000'000 * q.elapsed_ms), q.quality/q.elapsed_ms, q.passes_per_thread);
	}

	/*auto render_jpeg = [&](const rfkt::flame& f, std::string_view path, uint2 dims, double t, const rfkt::flame_kernel::bailout_args& bo) -> bool {

		auto k = fc.get_flame_kernel(fdb, rfkt::precision::f32, f);

		if (!k.kernel) {
			SPDLOG_INFO("\n{}\n{}", k.source, k.log);
			return false;
		}

		std::vector<double> samples = {};
		auto packer = [&samples](double v) { samples.push_back(v); };
		f.pack_samples(packer, invoker, t, 0, 4, pp.input_dims().x, pp.input_dims().y);

		auto state = k.kernel->warmup(stream, samples, std::move(bins), 0xDEADBEEF, 100);
		auto q = k.kernel->bin(stream, state, { .millis = 1000, .quality = 2000 }).get();
		auto res = pp.post_process(state.bins, out, q.quality, f.gamma.sample(t, invoker), f.brightness.sample(t, invoker), f.vibrancy.sample(t, invoker), true, stream).get();

	};*/

	

}

/*
	for (const auto& fname : rfkt::fs::list("assets/flames_test/", rfkt::fs::filter::has_extension(".flam3"))) {
	//{
	//	const std::filesystem::path fname = "assets/flames_test/electricsheep.247.34565.flam3";
		auto fxml = rfkt::fs::read_string(fname);
		auto f = rfkt::import_flam3(fdb, fxml);

		if (!f) {
			SPDLOG_ERROR("failed to import flame");
			return -1;
		}

		auto k = fc.get_flame_kernel(fdb, rfkt::precision::f32, *f);

		if (!k.kernel) {
			SPDLOG_INFO("\n{}\n{}", k.source, k.log);
			return -1;
		}

		std::vector<double> samples = {};
		auto packer = [&samples](double v) { samples.push_back(v); };
		f->pack_samples(packer, invoker, t, 0, 4, pp.input_dims().x, pp.input_dims().y);

		auto state = k.kernel->warmup(stream, samples, std::move(bins), 0xDEADBEEF, 100);
		auto q = k.kernel->bin(stream, state, { .millis = 1000, .quality = 2000 }).get();
		auto res = pp.post_process(state.bins, out, q.quality, f->gamma.sample(t, invoker), f->brightness.sample(t, invoker), f->vibrancy.sample(t, invoker), true, stream).get();


		SPDLOG_INFO("{}: {:.5f}, {:.5f} miters/ms {:.5f} mdraws/ms, {:.5f} ppt, {:.2f} kernel load", fname.filename().string(), q.quality, q.total_passes / (1'000'000 * q.elapsed_ms), q.total_draws / (1'000'000 * q.elapsed_ms), q.passes_per_thread, k.compile_ms);

		auto vec = je.encode_image(out.ptr(), pp.output_dims().x, pp.output_dims().y, 100, stream).get()();

		auto row = std::format("{},{},{},{}\n", fname.filename().string(), q.quality, q.total_passes / (1'000'000 * q.elapsed_ms), q.total_draws / (1'000'000 * q.elapsed_ms));
		rfkt::fs::write("stats.csv", row, true);

		auto basename = fname.filename();
		rfkt::fs::write("testrender/" + basename.string() + ".jpg", (const char*)vec.data(), vec.size(), false);
		
		bins = std::move(state.bins);
		bins.clear(stream);
	}

	return 0;
}
*/