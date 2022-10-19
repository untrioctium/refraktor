#include <App.h>
#include "concurrencpp/concurrencpp.h"

#include <vector_types.h>


#include <fmt/format.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util/nvenc.h>
#include <librefrakt/util/string.h>
#include <librefrakt/util/nvjpeg.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util.h>

#include <ranges>

auto cudaize(rfkt::cuda::context ctx, auto&& func) {
	return [ctx, func = std::move(func)]() {
		ctx.make_current_if_not();
		func();
	};
}

class event_loop_executor : public concurrencpp::derivable_executor<event_loop_executor> {
public:

	event_loop_executor(std::string_view name): 
		derivable_executor<event_loop_executor>("event_loop_executor"),
		loop(uWS::Loop::get()) {}

	void enqueue(concurrencpp::task task) override {
		loop->defer(std::move(task));
	}

	void enqueue(std::span<concurrencpp::task> tasks) override {
		for (auto& task : tasks) {
			loop->defer(std::move(task));
		}
	}

	int max_concurrency_level() const noexcept override {
		return 1;
	}

	bool shutdown_requested() const noexcept override {
		return false;
	}

	void shutdown() noexcept override {}
private:

	uWS::Loop* loop;
};
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
				1.0f / (static_cast<float>(quality) * sqrtf(10.0f)),
				static_cast<float>(brightness),
				static_cast<float>(vibrancy)
			));

		}
	private:
		cuda_module tm;
	};
}

class render_session : std::enable_shared_from_this<render_session> {

private:

	rfkt::flame flame_;
	rfkt::flame kernel_;

	std::optional<std::thread> daemon;


};

template<typename Type>
auto get_or_default(nlohmann::json& js, std::string name, Type&& def) {
	auto& v = js[name];
	if (v.is_null()) return def;
	if (std::is_arithmetic_v<Type> && !v.is_number()) return def;
	if (std::is_same_v<std::string, Type> && !v.is_string()) return def;
	return v.get<Type>();
}

template<size_t N>
struct StringLiteral {
	constexpr StringLiteral(const char(&str)[N]) {
		std::copy_n(str, N, value);
	}

	char value[N];
};

template<StringLiteral code, StringLiteral desc>
void end_error(uWS::HttpResponse<true>* res, uWS::HttpRequest* req = nullptr) {
	res->writeStatus(fmt::format("{} {}", code.value, desc.value));
	res->end();

	if (req) SPDLOG_INFO("{} {} {}", code.value, req->getMethod(), req->getUrl());
	else SPDLOG_INFO("{}", code.value);
}

class query_parser {
public:
	enum arg_type {
		integer,
		decimal,
		string,
		boolean
	};

	query_parser(std::initializer_list<std::pair<std::string, arg_type>> defs) {
		for (const auto& entry : defs) {
			args[entry.first] = entry.second;
		}
	}

	auto parse(std::string_view data) const -> nlohmann::json {
		auto ret = nlohmann::json::object();

		for (auto& [name, type] : args) {
			if (type == boolean) ret[name] = false;
		}

		for (const auto str : std::ranges::views::split(data, '&')) {
			auto p = std::string_view{ str.begin(), str.end() };
			auto key = std::string{ p.substr(0, p.find('=')) };
			auto val = std::string{ p.substr(p.find('=') + 1) };

			if (!args.contains(key)) continue;
			switch (args.at(key)) {
			case integer:
				if (val.find_first_not_of("1234567890") != std::string::npos) continue;
				ret[key] = std::stol(val);
				break;

			case decimal:
				if (val.find_first_not_of("1234567890.") != std::string::npos) continue;
				ret[key] = std::stod(val);
				break;

			case string:
				ret[key] = val;
				break;

			case boolean:
				ret[key] = true;
				break;

			default: break;
			}
		}

		return ret;
	}

private:
	std::map< std::string, arg_type, std::less<>> args;
};

class http_connection_data : public std::enable_shared_from_this<http_connection_data> {
public:
	http_connection_data() = delete;
	~http_connection_data() {
		SPDLOG_INFO("Closing HTTP session {}", fmt::ptr(this));
	}

	http_connection_data(const http_connection_data&) = delete;
	http_connection_data& operator=(const http_connection_data&) = delete;

	http_connection_data(http_connection_data&&) = delete;
	http_connection_data& operator=(const http_connection_data&&) = delete;

	[[nodiscard]] static auto make(auto* res) {
		auto ptr = std::shared_ptr<http_connection_data>{ new http_connection_data{res} };
		ptr->response()->onAborted([cd = ptr->shared_from_this()]() { SPDLOG_INFO("Aborting HTTP session {}", fmt::ptr(cd.get()));  cd->abort(); });
		return ptr;
	}

	auto response() { return response_; }
	void abort() { aborted_ = true; }
	auto aborted() { return aborted_.operator bool(); }

	explicit operator bool() { return aborted(); }

	void defer(std::move_only_function<void(http_connection_data&)>&& func) {
		parent->defer([cd = shared_from_this(), func = std::move(func)]() mutable {
			func(*cd);
		});
	}
private:

	uWS::HttpResponse<true>* response_;
	std::atomic_bool aborted_ = false;
	uWS::Loop* parent;

	http_connection_data(uWS::HttpResponse<true>* res) : response_(res), parent(uWS::Loop::get()) {
		SPDLOG_INFO("Opening HTTP session {}", fmt::ptr(this));
	}

};

struct socket_data  {
	rfkt::flame flame;
	rfkt::flame_kernel kernel;

	std::size_t total_frames;
	double loops_per_frame;
	int fps;
	uint2 dims;

	std::unique_ptr<rfkt::nvenc::session> session;
	std::shared_ptr<rfkt::cuda_buffer<uchar4>> render;

	std::atomic_bool closed;
	std::move_only_function<void(std::vector<std::byte>&&, std::shared_ptr<socket_data>)> feed;

	~socket_data() {
		SPDLOG_INFO("Closing socket");
	}
};

auto session_render_thread(std::shared_ptr<socket_data> ud, const rfkt::tonemapper& tm, rfkt::cuda::context ctx) {
	return cudaize(ctx, [ud = move(ud), &tm, ctx]() {
		auto tls = rfkt::cuda::thread_local_stream();

		std::size_t sent_bytes = 0;

		auto time_since_start = [start = std::chrono::high_resolution_clock::now()]() {
			return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1'000'000'000.0;
		};

		auto slept_time = 0.0;
		auto pre_fut_time = 0.0;
		auto wait_time = 0.0;
		auto encode_time = 0.0;
		std::size_t total_draws = 0;

		while (!ud->closed) {

			auto t = ud->total_frames * ud->loops_per_frame;

			auto pre_fut_start = time_since_start();
			auto state = ud->kernel.warmup(
				tls,
				ud->flame,
				ud->dims,
				t,
				1, ud->loops_per_frame, 0xdeadbeef, 64
			);

			auto result_fut = ud->kernel.bin(tls, state, 1000, 1000 / ud->fps - 5, 1'000'000'000);

			auto gamma = ud->flame.gamma.sample(t);
			auto brightness = ud->flame.brightness.sample(t);
			auto vibrancy = ud->flame.vibrancy.sample(t);

			// while we wait, let's encode the last frame
			if (ud->total_frames > 0) {
				auto encode_start = time_since_start();
				auto ret = ud->session->submit_frame((ud->total_frames - 1) % (ud->fps) == 0);
				encode_time += time_since_start() - encode_start;

				if (ret) {
					ret->insert(ret->begin(), std::byte{ 0 });
					sent_bytes += ret->size();
					ud->feed(std::move(ret.value()), ud);
				}
			}
			pre_fut_time += time_since_start() - pre_fut_start;
			auto wait_start = time_since_start();
			auto result = result_fut.get();
			total_draws += result.total_draws;
			wait_time += time_since_start() - wait_start;

			tm.run(state.bins.ptr(), ud->render->ptr(), ud->dims, result.quality, false, gamma, brightness, vibrancy, tls);
			cuStreamSynchronize(rfkt::cuda::thread_local_stream());


			ud->total_frames++;

			double avg_quality = (total_draws / double(ud->dims.x * ud->dims.y)) / ud->total_frames;
			if (ud->total_frames > 0 && ud->total_frames % ud->fps == 0) {

				double factor = 1000.0 / ud->total_frames;

				double draws_per_ms = total_draws / (1000 * (time_since_start() - slept_time)) / 1'000'000;
				double real_fps = ud->total_frames / time_since_start();

				SPDLOG_INFO("{:.4} MB, {:.3} mbps, {:.3} avg quality, {:.3}m draws/ms, {:.3} ms/frame avg, {:.3} ms/frame to future get, {:.3} ms/frame future wait. {:.4} real fps",
					sent_bytes / (1024.0 * 1024.0),
					(8.0 * sent_bytes / (1024.0 * 1024.0)) / (time_since_start()),
					avg_quality,
					draws_per_ms,
					(time_since_start() - slept_time) * factor,
					pre_fut_time * factor,
					wait_time * factor,
					real_fps);
			}

			if (result.quality < avg_quality * .8) {
				SPDLOG_INFO("Dropped frame detected, {} quality", result.quality);
			}

			if (ud->total_frames / double(ud->fps) - time_since_start() > .2) {
				SPDLOG_INFO("Sleeping, overbuffered");
				auto sleep = 200 - 1000 / ud->fps * 2;
				slept_time += sleep / 1000;
				std::this_thread::sleep_for(std::chrono::milliseconds{ sleep });
			}
		}
	});
}


int main(int argc, char** argv) {

	SPDLOG_INFO("Starting refrakt-server");
	rfkt::flame_info::initialize("config/variations.yml");

	SPDLOG_INFO("Flame system initialized: {} variations, {} parameters", rfkt::flame_info::num_variations(), rfkt::flame_info::num_parameters());

	
	auto ctx = rfkt::cuda::init();
	auto dev = ctx.device();

	SPDLOG_INFO("Using device {}, CUDA {}.{}", dev.name(), dev.compute_major(), dev.compute_minor());
	
	concurrencpp::runtime runtime;

	auto km = rfkt::kernel_manager{};
	auto fc = rfkt::flame_compiler{ km };
	auto tm = rfkt::tonemapper{ km };
	auto jpeg = rfkt::nvjpeg::encoder{ rfkt::cuda::thread_local_stream() };

	auto [sm_result, sm] = km.compile_file("assets/kernels/smooth.cu",
		rfkt::compile_opts("smooth")
		.function("smooth")
		.flag(rfkt::compile_flag::extra_vectorization)
		.flag(rfkt::compile_flag::use_fast_math)
	);

	if (!sm_result.success) {
		SPDLOG_ERROR("{}\n", sm_result.log);
		exit(1);
	}
	auto func = sm();
	auto [ss_grid, ss_block] = func.suggested_dims();
	SPDLOG_INFO("Loaded smooth kernel: {} regs, {} shared, {} local, {}x{} suggested dims", func.register_count(), func.shared_bytes(), func.local_bytes(), ss_grid, ss_block);

	auto jpeg_executor = runtime.make_worker_thread_executor();
	jpeg_executor->post([ctx]() {
		ctx.make_current();
	});

	auto app = uWS::SSLApp();

	uWS::Loop::get()->defer([ctx]() {
		ctx.make_current_if_not();
	});

	auto local_flames = rfkt::fs::list("assets/flames", rfkt::fs::filter::has_extension(".flam3"));

	// yeah yeah yeah, rand is bad. not cryptographically securing bank transactions here
	srand(time(0));

	using ws_t = uWS::WebSocket<true, true, std::shared_ptr<socket_data>>;
	app.ws<std::shared_ptr<socket_data>>("/stream", {
		.compression = uWS::DISABLED,
		.maxPayloadLength = 100 * 1024 * 1024,
		.idleTimeout = 16,
		.maxBackpressure = 100 * 1024 * 1024,
		.closeOnBackpressureLimit = false,
		.resetIdleTimeoutOnSend = false,
		.sendPingsAutomatically = true,
		.upgrade = nullptr,

		.open = [](ws_t* ws) {
			SPDLOG_INFO("Opened session for streaming");
		},
		.message = [&fc, &tm, &runtime, ctx, &local_flames](ws_t* ws, std::string_view message, uWS::OpCode opCode) {
			auto* ud = ws->getUserData();
			auto js = nlohmann::json::parse(message);
			if (!js.is_object()) return;
			auto& cmd_node = js["cmd"];
			if (!cmd_node.is_string()) return;

			std::string cmd = js["cmd"].get<std::string>();
			if (cmd == "begin") {
				auto flame = (js.count("flame") != 0) ? fmt::format("assets/flames/electricsheep.247.{}.flam3", js["count"].get<std::string>()) : local_flames[rand() % local_flames.size()].string();

				auto width = get_or_default(js, "width", 1280u);
				auto height = get_or_default(js, "height", 720u);
				auto seconds_per_loop = get_or_default(js, "loop_length", 5.0);
				auto fps = get_or_default(js, "fps", 30);

				auto f = rfkt::flame::import_flam3(flame);
				if (!f) return;

				// add the tiniest bit of rotation to show dynamic nature
				//f->rotate.ani = rfkt::animator::make("increase", nlohmann::json::object({ 
				//	{"per_loop",-360.0 / (60/ seconds_per_loop)}
				//}));

				auto k_result = fc.get_flame_kernel(rfkt::precision::f32, f.value());
				if (!k_result.kernel.has_value()) return;

				auto sesh = rfkt::nvenc::session::make();
				auto buf = sesh->initialize({ width, height }, { fps, 1 });

				auto new_ptr = std::shared_ptr<socket_data>{ new socket_data{
					.flame = std::move(f.value()),
					.kernel = std::move(*k_result.kernel),
					.total_frames = 0,
					.loops_per_frame = 1.0 / (seconds_per_loop * fps),
					.fps = fps,
					.dims = {width, height},
					.session = std::move(sesh),
					.render = std::move(buf),
					.closed = false,
					.feed = [loop = uWS::Loop::get(), ws](std::vector<std::byte>&& d, std::shared_ptr<socket_data> ud) mutable {
						loop->defer([ws, d = std::move(d), ud = std::move(ud)]() mutable {
							 if(!ud->closed) ws->send(std::string_view{(const char*)d.data(), d.size()});
						});
					}
				} };

				ud->swap(new_ptr);

				runtime.thread_executor()->post(session_render_thread(*ud, tm, ctx));

				SPDLOG_INFO("Starting session: {}, {}x{}, {} fps", flame, width, height, fps);
			}
			else if (cmd == "mod" && *ud) {
				auto path = get_or_default<std::string>(js, "path", "");
				if (path.empty()) return;
				auto val = get_or_default(js, "val", std::numeric_limits<double>::quiet_NaN());
				if (std::isnan(val)) return;

				auto ptr = ud->get()->flame.seek(path);
				if (!ptr) return;
				ptr->t0 = val;
			}
			else if (cmd == "dump" && *ud) {
				auto dump = (*ud)->flame.dump();
				uWS::Loop::get()->defer([dump = std::move(dump), ws]() {
					ws->send(dump);
				});
			}
		},
		.drain = [](auto*/*ws*/) {
			/* Check ws->getBufferedAmount() here */
		},
		.ping = [](auto*/*ws*/, std::string_view) {
			/* Not implemented yet */
		},
		.pong = [](auto*/*ws*/, std::string_view) {
			/* Not implemented yet */
		},
		.close = [](ws_t* ws, int /*code*/, std::string_view /*message*/) {
			if(*ws->getUserData()) ws->getUserData()->get()->closed = true;
			/* You may access ws->getUserData() here */
		}
	});

	app.get("/bananas", [](auto* res, auto* req) {
		auto page = rfkt::fs::read_string("assets/static/muxer.html");

		res->writeStatus("200 OK");
		res->tryEnd(page);
	});

	app.get("/render/:id", [&](auto* res, auto* req) {

		SPDLOG_INFO("get {}?{}", req->getUrl(), req->getQuery());
		static const query_parser qp{
			{"width", query_parser::integer},
			{"height", query_parser::integer},
			{"quality", query_parser::decimal},
			{"time", query_parser::decimal},
			{"jpeg_quality", query_parser::integer},
			{"fps", query_parser::integer},
			{"loop_speed", query_parser::integer },
			{"bin_time", query_parser::integer}

		};

		auto data = qp.parse(req->getQuery());

		struct render_data {
			unsigned int width, height, fps, loop_speed, jpeg_quality, bin_time;
			double time, quality;
		} rd{};

		rd.width = data.value("width", 1920);
		rd.height = data.value("height", 1080);
		rd.quality = data.value("quality", 128.0f);
		rd.time = data.value("time", 0.0f);
		rd.jpeg_quality = data.value("jpeg_quality", 100);
		rd.fps = data.value("fps", 30);
		rd.loop_speed = data.value("loop_speed", 5000);
		rd.bin_time = data.value("bin_time", 2000);

		if (rd.width > 3840) rd.width = 3840;
		if (rd.height > 2160) rd.height = 2160;
		if (rd.quality > 2048) rd.quality = 2048;
		if (rd.bin_time > 2000) rd.bin_time = 2000;
		if (rd.jpeg_quality <= 0 || rd.jpeg_quality > 100) rd.jpeg_quality = 85;

		auto cd = http_connection_data::make(res);
		auto flame_path = fmt::format("assets/flames/electricsheep.247.{}.flam3", req->getParameter(0));

		if (!rfkt::fs::exists(flame_path)) {
			end_error<"404", "Not Found">(res, req);
			return;
		}

		jpeg_executor->post([flame_path = std::move(flame_path), cd = std::move(cd), rd = std::move(rd), ctx = ctx, &fc, &tm, &jpeg, &sm]() mutable {
			auto tls = rfkt::cuda::thread_local_stream();

			auto timer = rfkt::timer();
			auto fopt = rfkt::flame::import_flam3(flame_path);
			auto t_load = timer.count();

			if (!fopt) {
				if(!cd->aborted()) end_error<"500", "Internal Server Error">(cd->response());
				return;
			}

			timer.reset();
			auto k_result = fc.get_flame_kernel(rfkt::precision::f32, fopt.value());
			auto t_kernel = timer.count();

			if (!k_result.kernel.has_value()) {
				if (!cd->aborted()) end_error<"500", "Internal Server Error>">(cd->response());
				return;
			}

			SPDLOG_INFO("{}", k_result.source);


			auto kernel = rfkt::flame_kernel{ std::move(k_result.kernel.value()) };
			auto render = rfkt::cuda::make_buffer_async<uchar4>(rd.width * rd.height, tls);
			auto state = kernel.warmup(tls, fopt.value(), { rd.width, rd.height }, rd.time, 1, double(rd.fps) / rd.loop_speed, 0xdeadbeef, 100);

			timer.reset();
			auto result = kernel.bin(tls, state, rd.quality, rd.bin_time, 1'000'000'000).get();
			auto t_bin = timer.count();

			auto smoothed = rfkt::cuda::make_buffer_async<float4>(rd.width * rd.height, tls);
			cuMemsetD32Async(smoothed.ptr(), 0, rd.width* rd.height * 4, tls);

			sm().launch({ rd.width / 8 + 1, rd.height / 8 + 1, 1 }, { 8,8,1 }, tls)(
				state.bins.ptr(),
				smoothed.ptr(),
				rd.width, rd.height,
				(int) 11,
				(int) 0,
				0.6f
				);

			SPDLOG_INFO("load {}, kernel {}, bin {}, quality {}", t_load, t_kernel, t_bin, result.quality);

			tm.run(smoothed.ptr(), render.ptr(), { rd.width, rd.height }, result.quality, true, fopt->gamma.sample(rd.time), fopt->brightness.sample(rd.time), fopt->vibrancy.sample(rd.time), tls);
			auto data = jpeg.encode_image(render.ptr(), rd.width, rd.height, rd.jpeg_quality, tls).get();
 
			cd->defer([data = std::move(data)](auto& cd){
				if (!cd.aborted()) {
					cd.response()->writeHeader("Content-Type", "image/jpeg");
					cd.response()->tryEnd({(char*)data.data(), data.size()});
				}
			});
		});

	});

	app.listen(3000,
		[](auto* listen_socket) {
			if (listen_socket) {
				SPDLOG_INFO("Listening on port 3000");
			}
		});

	app.run();

	return 1;
}