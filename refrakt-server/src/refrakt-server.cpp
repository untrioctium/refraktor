#include <App.h>
#include "concurrencpp/concurrencpp.h"

#include <vector_types.h>


#include <fmt/format.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/util/nvenc.h>

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
				.function("tonemap")
				.flag(rfkt::compile_flag::extra_vectorization)
				.flag(rfkt::compile_flag::use_fast_math)
			);

			if (!tm_result.success) {
				SPDLOG_ERROR("{}", tm_result.log);
				exit(1);
			}

			tm = std::move(mod);
		}

		void run(CUdeviceptr bins, CUdeviceptr out, uint2 dims, float quality, const flame& f) {

			CUDA_SAFE_CALL(tm.kernel().launch({ dims.x / 8 + 1, dims.y / 8 + 1, 1 }, { 8, 8, 1 }, rfkt::cuda::thread_local_stream())(
				bins,
				out,
				dims.x, dims.y,
				static_cast<float>(f.gamma.sample(0)),
				1.0f / (quality * sqrtf(10.0f)),
				static_cast<float>(f.brightness.sample(0)),
				static_cast<float>(f.vibrancy.sample(0))
			));

		}
	private:
		cuda_module tm;
	};
}

template<typename Type>
auto get_or_default(nlohmann::json& js, std::string name, Type&& def) {
	auto& v = js[name];
	if (v.is_null()) return def;
	if (std::is_arithmetic_v<Type> && !v.is_number()) return def;
	if (std::is_same_v<std::string, Type> && !v.is_string()) return def;
	return v.get<Type>();
}

int main(int argc, char** argv) {

	SPDLOG_INFO("Starting refrakt-server");
	rfkt::flame_info::initialize("config/variations.yml");

	
	auto ctx = rfkt::cuda::init();
	auto dev = ctx.device();

	SPDLOG_INFO("Using device {}, CUDA {}.{}", dev.name(), dev.compute_major(), dev.compute_minor());
	
	concurrencpp::runtime runtime;

	auto km = rfkt::kernel_manager{};
	auto fc = rfkt::flame_compiler{ km };
	auto tm = rfkt::tonemapper{ km };

	auto app = uWS::SSLApp();

	uWS::Loop::get()->defer([ctx]() {
		ctx.make_current_if_not();
	});

	struct socket_data {
		rfkt::flame flame;
		rfkt::flame_kernel kernel;

		std::size_t total_frames;
		double loops_per_frame;
		int fps;
		uint2 dims;

		std::unique_ptr<rfkt::nvenc::session> session;
		std::shared_ptr<rfkt::cuda_buffer<uchar4>> render;

		std::atomic_bool closed;
		uWS::MoveOnlyFunction<void(std::vector<std::byte>&&)> feed;

		~socket_data() {
			SPDLOG_INFO("Closing socket");
		}
	};
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
		.message = [&km, &fc, &tm, &runtime](ws_t* ws, std::string_view message, uWS::OpCode opCode) {
			auto* ud = ws->getUserData();
			auto js = nlohmann::json::parse(message);
			if (!js.is_object()) return;
			auto& cmd_node = js["cmd"];
			if (!cmd_node.is_string()) return;

			std::string cmd = js["cmd"].get<std::string>();
			if (cmd == "begin") {
				auto flame = get_or_default<std::string>(js, "flame", "53476");
	

				auto width = get_or_default(js, "width", 1280u);
				auto height = get_or_default(js, "height", 720u);
				auto seconds_per_loop = get_or_default(js, "loop_length", 5.0);
				auto fps = get_or_default(js, "fps", 30);

				auto f = rfkt::flame::import_flam3(fmt::format("assets/flames/electricsheep.247.{}.flam3", flame));
				if (!f) return;

				// add the tiniest bit of rotation to show dynamic nature
				f->rotate.ani = rfkt::animator::make("increase", nlohmann::json::object({ 
					{"per_loop",360.0 / (60/ seconds_per_loop)}
				}));

				auto k_result = fc.get_flame_kernel(rfkt::precision::f32, f.value());
				if (!k_result.kernel.has_value()) return;

				auto sesh = rfkt::nvenc::session::make();
				auto buf = sesh->initialize({ width, height }, { fps, 1 });

				auto new_ptr = std::shared_ptr<socket_data>{ new socket_data{
					.flame = std::move(f.value()),
					.kernel = std::move(k_result.kernel.value()),
					.total_frames = 0,
					.loops_per_frame = 1.0 / (seconds_per_loop * fps),
					.fps = fps,
					.dims = {width, height},
					.session = std::move(sesh),
					.render = std::move(buf),
					.closed = false,
					.feed = [loop = uWS::Loop::get(), ws](std::vector<std::byte>&& d) {
						loop->defer([ws, d = std::move(d)]() {
							ws->send(std::string_view{(const char*)d.data(), d.size()});
						});
					}
				} };

				ud->swap(new_ptr);

				runtime.thread_executor()->post([&tm, ud = *ud, ctx = rfkt::cuda::context::current()]() {
					ctx.make_current_if_not();
					auto tls = rfkt::cuda::thread_local_stream();

					std::size_t sent_bytes = 0;
					
					auto time_since_start = [start = std::chrono::high_resolution_clock::now()]() {
						return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1'000'000'000.0;
					};

					auto slept_time = 0.0;
					auto pre_fut_time = 0.0;
					auto wait_time = 0.0;

					while (!ud->closed) {

						auto pre_fut_start = time_since_start();
						auto state = ud->kernel.warmup(
							tls,
							ud->flame,
							ud->dims,
							ud->total_frames * ud->loops_per_frame,
							1, ud->loops_per_frame, 0xdeadbeef, 64
						);

						auto result_fut = ud->kernel.bin(tls, state, 10000, 26, 1'000'000'000);

						// while we wait, let's encode the last frame
						if (ud->total_frames > 0){
							auto ret = ud->session->submit_frame(ud->total_frames % (ud->fps * 5) == 0);

							if (ret) {
								sent_bytes += ret->size();
								ud->feed(std::move(ret.value()));
							}
						}
						pre_fut_time += time_since_start() - pre_fut_start;
						auto wait_start = time_since_start();
						auto result = result_fut.get();
						wait_time += time_since_start() - wait_start;

						tm.run(state.bins.ptr(), ud->render->ptr(), ud->dims, result.quality, ud->flame);
						cuStreamSynchronize(rfkt::cuda::thread_local_stream());


						ud->total_frames++;

						if (ud->total_frames > 0 && ud->total_frames % ud->fps == 0) {
							double factor = 1000.0 / ud->total_frames;
							SPDLOG_INFO("{:.4} MB, {:.3} mbps, {:.3} ms/frame avg, {:.3} ms/frame to future get, {:.3} ms/frame future wait",
								sent_bytes / (1024.0 * 1024.0),
								(8.0 * sent_bytes / (1024.0 * 1024.0)) / (time_since_start()),
								(time_since_start() - slept_time) * factor,
								pre_fut_time * factor,
								wait_time * factor);
						}

						if (ud->total_frames / double(ud->fps) - time_since_start() > 1.0) {
							SPDLOG_INFO("Sleeping, overbuffered");
							slept_time += .5;
							std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
						}
					}

				});

				SPDLOG_INFO("Starting session: {}, {}x{}, {} fps", flame, width, height, fps);
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

	app.listen(3000,
		[](auto* listen_socket) {
			if (listen_socket) {
				SPDLOG_INFO("Listening on port 3000");
			}
		});

	app.run();

	return 1;
}