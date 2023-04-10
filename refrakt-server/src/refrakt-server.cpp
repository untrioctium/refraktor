#include <App.h>
#include "concurrencpp/concurrencpp.h"

#include <vector_types.h>

#include <librefrakt/flame_compiler.h>
#include <eznve.hpp>
#include <librefrakt/util/nvjpeg.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util.h>

#include <librefrakt/image/tonemapper.h>
#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/converter.h>

#include <librefrakt/util/gpuinfo.h>

#include <cmath>
#include <ranges>

#include <readerwritercircularbuffer.h>

using json = nlohmann::json;

std::vector<rfkt::flame> gen_samples(const rfkt::flame& f, double t, double loops_per_frame) {
	std::vector<rfkt::flame> samples{};
	for (int i = 0; i < 4; i++) {
		auto sample = f;
		for (auto& x : sample.xforms) {
			for (auto& vl : x.vchain) {
				if (vl.per_loop > 0) {
					const auto base = t * vl.per_loop;
					vl.transform = vl.transform.rotated(base + vl.per_loop * loops_per_frame * (i - 1));
				}
			}
		}
		samples.emplace_back(std::move(sample));
	}
	return samples;
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

		for (const auto& str : std::ranges::views::split(data, '&')) {
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

namespace rfkt {
	class postprocessor {
	public:
		postprocessor(ezrtc::compiler& kc, uint2 dims, bool upscale) :
			tm(kc),
			dn(dims, upscale),
			conv(kc),
			tonemapped((upscale) ? (dims.x * dims.y / 4) : (dims.x * dims.y)),
			denoised(dims.x* dims.y),
			dims_(dims),
			upscale(upscale)
		{

		}

		~postprocessor() = default;

		postprocessor(const postprocessor&) = delete;
		postprocessor& operator=(const postprocessor&) = delete;

		postprocessor(postprocessor&&) = default;
		postprocessor& operator=(postprocessor&&) = default;

		auto make_output_buffer() const -> rfkt::cuda_buffer<uchar4> {
			return { dims_.x * dims_.y };
		}

		auto make_output_buffer(CUstream stream) const -> rfkt::cuda_buffer<uchar4> {
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
			rfkt::cuda_view<float4> in,
			rfkt::cuda_view<uchar4> out,
			double quality,
			double gamma, double brightness, double vibrancy,
			bool planar_output,
			cuda_stream& stream) {

			auto promise = std::promise<double>{};
			auto future = promise.get_future();

			stream.host_func(
				[&t = perf_timer]() mutable {
				t.reset();
				});

			tm.run(in, tonemapped, input_dims(), quality, gamma, brightness, vibrancy, stream);
			dn.denoise(output_dims(), tonemapped, denoised, stream);
			conv.to_24bit(denoised, out, output_dims(), planar_output, stream);
			stream.host_func([&t = perf_timer, p = std::move(promise)]() mutable {
				p.set_value(t.count());
			});

			return future;
		}

	private:
		rfkt::tonemapper tm;
		rfkt::denoiser dn;
		rfkt::converter conv;

		rfkt::cuda_buffer<half3> tonemapped;
		rfkt::cuda_buffer<half3> denoised;

		rfkt::timer perf_timer;

		uint2 dims_;
		bool upscale;
	};
}

namespace rfkt {
	class render_socket : public std::enable_shared_from_this<render_socket> {
	public:

		using handle = std::shared_ptr<render_socket>;
		using send_function = std::move_only_function<void(handle&&, std::vector<char>&&)>;

		[[nodiscard]] static handle make(rfkt::flame_compiler& fc, ezrtc::compiler& km, rfkt::flamedb& fdb, rfkt::cuda::context ctx, concurrencpp::runtime& rt, send_function&& sf) {
			auto ret = handle{ new render_socket{fc, km, fdb} };
			ret->send = std::move(sf);
			ret->render_queue = rt.make_worker_thread_executor();
			ret->work_queue = rt.thread_pool_executor();
			ret->timer_queue = rt.timer_queue();
			ret->ctx = ctx;

			ret->render_queue->enqueue([ctx]() {
				ctx.make_current_if_not();
			});

			return ret;
		}

		void on_message(std::string_view data) {
			auto cmd = json::parse(data, nullptr, false);

			if (cmd.is_discarded()) {
				SPDLOG_WARN("could not parse json: {}", data);
				return;
			}

			if (not cmd.contains("cmd") or not cmd["cmd"].is_string()) {
				SPDLOG_WARN("cmd is not a string");
				return;
			}

			if (cmd["cmd"] == "begin") {
				on_begin(std::move(cmd));
			}
		}

		void on_close() {
			is_closed = true;
			if (send_timer) {
				send_timer->cancel();
			}
		}

		[[nodiscard]] bool closed() const noexcept {
			return is_closed;
		}

		~render_socket() {
			SPDLOG_INFO("closed socket");
		}

	private:

		struct last_frame {
			rfkt::cuda_buffer<float4> bins = {};
			rfkt::cuda_buffer<ushort4> accumulator = {};
			double quality = 0.0;
			double gamma = 0.0;
			double brightness = 0.0;
			double vibrancy = 0.0;
			rfkt::cuda_stream pp_stream{};

			std::size_t total_passes = 0;
			std::size_t total_draws = 0;
			double total_time = 0.0;
		};

		render_socket(rfkt::flame_compiler& fc, ezrtc::compiler& km, rfkt::flamedb& fdb) :
			chunks(10),
			fc(fc),
			km(km),
			fdb(fdb) {}

		void on_begin(json&& data) {

			auto flame_path = [&]() -> rfkt::fs::path {
				if (not data.contains("flame")) {
					auto local_flames = rfkt::fs::list("assets/flames_test", rfkt::fs::filter::has_extension(".flam3"));
					return local_flames[std::rand() % local_flames.size()];
				}
				return fmt::format("assets/flames_test/electricsheep.{}.flam3", data["flame"].get<std::string>());
			}();

			auto upscale = data.value("upscale", false);
			auto width = data.value("width", 1280u);
			auto height = data.value("height", 720u);
			auto seconds_per_loop = data.value("loop_length", 5.0);
			this->fps = data.value("fps", 30u);
			this->loops_per_frame = 1.0 / (seconds_per_loop * fps);

			this->max_bin_time = 1000.0 / fps - 2;

			this->flame = rfkt::import_flam3(fdb, rfkt::fs::read_string(flame_path));
			if (not flame) {
				SPDLOG_WARN("could not import {}", flame_path.string());
				return;
			}

			// add a tiny bit of rotation to each xform
			if (data.value("embellish", false)) {
				for(auto& xf: flame->xforms) {
					//if (id == -1) return;

					for (auto& vlink : xf.vchain) {
						if (vlink.per_loop == 0) {
							vlink.per_loop = 5 * (rand() & 1) ? -1 : 1;
						}
					}
				}
			}

			auto k_result = fc.get_flame_kernel(fdb, rfkt::precision::f32, flame.value());

			SPDLOG_ERROR("\n{}\n---------------\n{}\n=====\n{}\n", flame_path.string(), k_result.source, k_result.log);

			if (not k_result.kernel.has_value()) {

				return;
			}

			this->kernel = std::move(k_result.kernel);

			this->pp = rfkt::postprocessor(km, { width, height }, upscale);
			this->encoder = std::make_unique<eznve::encoder>( this->pp->output_dims(), uint2{this->fps, 1}, eznve::codec::h264, ctx );

			this->render_queue->enqueue([self = shared_from_this()]() mutable {
				self->start = std::chrono::high_resolution_clock::now();
				render_frame(std::move(self), std::nullopt);
			});

			this->send_timer = this->timer_queue->make_timer(std::chrono::milliseconds(0), std::chrono::milliseconds(static_cast<long long>(std::floor(1000.0 / this->fps) - 5)), this->work_queue, [self = shared_from_this()]() {
				if (self->total_frames % self->fps == 0) {
					SPDLOG_INFO("{} megabits/second", self->encoder->total_bytes() / self->secs_since_start() / (1'000'000) * 8);
				}
				self->send_frame(self->shared_from_this());
			});

		}

		static void send_frame(std::shared_ptr<render_socket>&& self) {
			auto data = std::vector<char>{};
			if (self->chunks.try_dequeue(data)) {
				auto& sender = self->send;
				sender(std::move(self), std::move(data));
			}
			else {
				SPDLOG_INFO("buffer miss");
			}
		}

		static void render_frame(std::shared_ptr<render_socket>&& self, std::optional<last_frame>&& lf) {
			if (self->is_closed) {
				SPDLOG_INFO("quitting render");
				return;
			}

			if (lf) {
				self->pp->post_process(lf->bins, { self->encoder->buffer(), self->encoder->buffer_size() }, lf->quality, lf->gamma, lf->brightness, lf->vibrancy, false, lf->pp_stream);
			}

			const auto t = self->total_frames * self->loops_per_frame;

			rfkt::timer frame_timer{};

			auto state = self->kernel->warmup(self->stream,
				gen_samples(self->flame.value(), t, self->loops_per_frame),
				self->pp->input_dims(),
				0xdeadbeef,
				64
			);

			auto frame_quality = 0.0;
			auto subpasses = 0;

			while (frame_quality < (.95 * self->target_quality.value_or(1000)) && subpasses < 2) {
				if (subpasses > 0) {
					SPDLOG_INFO("repairing frame ({}/10 buffered, wanted {:.4}, got {:.4})", self->chunks.size_approx(), self->target_quality.value(), frame_quality);
				}
				auto result = self->kernel->bin(self->stream, state, { .iters = 1'000'000, .millis = static_cast<unsigned int>(self->max_bin_time), .quality = self->target_quality.value_or(1000) - frame_quality }).get();

				if (self->total_frames % self->fps == 0) {
					SPDLOG_INFO("{} m draws/ms", (result.total_draws / 1'000'000) / result.elapsed_ms);
				}

				frame_quality += result.quality;
				if (lf) {
					lf->total_passes += result.total_passes;
					lf->total_draws += result.total_draws;
					lf->total_time += result.elapsed_ms;
				}
				if (not self->target_quality) self->target_quality = frame_quality;
				subpasses++;
			}

			if (self->target_quality.has_value() && frame_quality < self->target_quality.value() * .95 or subpasses > 1) {
				if (self->chunks.size_approx() <= self->chunks.max_capacity()) {
					if (self->chunks.size_approx() <= 4) self->target_quality.value() *= .975;
					else if (self->chunks.size_approx() <= 8) self->target_quality.value() *= .99;
					else self->target_quality.value() *= .995;
				}
				else {
					self->target_quality.value() *= .995;
				}
			}

			self->total_frames++;
			
			/*self->pp->post_process(state.bins, {self->encoder->buffer(), self->encoder->buffer_size()}, frame_quality,
				self->flame->gamma,
				self->flame->brightness,
				self->flame->vibrancy,
				false,
				self->stream).get();*/

			if (lf) {
				lf->pp_stream.sync();
				const auto flag = ((self->total_frames) % self->fps == 0) ? eznve::frame_flag::idr : eznve::frame_flag::none;
				auto chunk = self->encoder->submit_frame(flag);

				auto total_time = frame_timer.count();

				if (chunk) {
					if (not self->chunks.try_enqueue(chunk->data)) {
						if (total_time < 1000.0 / self->fps && subpasses == 1) self->target_quality.value() *= 1.01;
						while (not self->chunks.wait_enqueue_timed(chunk->data, 1000)) {
							if (self->closed()) return;
						};
					}
				}
			} 

			if (not lf) {
				lf = last_frame{};
			}

			lf->bins = std::move(state.bins);
			lf->quality = frame_quality;
			lf->gamma = self->flame->gamma;
			lf->brightness = self->flame->brightness;
			lf->vibrancy = self->flame->vibrancy;
			//SPDLOG_INFO("{}", frame_quality);
			if (self->total_frames % self->fps == 0 && self->total_frames > 0) {
				SPDLOG_INFO("buffer: {}/10, target: {:.4}", self->chunks.size_approx(), self->target_quality.value());
				SPDLOG_INFO("{:.4} billion iters/sec, {:.4} billion draws/sec", (lf->total_passes / 1'000'000'000.0) / self->secs_since_start(), (lf->total_draws / 1'000'000'000.0) / self->secs_since_start());
			}

			auto pool = self->render_queue.get();
			pool->enqueue([self = std::move(self), lf = std::move(lf)]() mutable {
				render_frame(std::move(self), std::move(lf));
			});
		}

		double secs_since_start() {
			return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1'000'000.0;
		}

		rfkt::cuda::context ctx;

		rfkt::cuda_stream stream;

		ezrtc::compiler& km;
		rfkt::flame_compiler& fc;
		std::optional<rfkt::flame> flame;
		std::optional<rfkt::flame_kernel> kernel;
		std::optional<rfkt::postprocessor> pp;
		std::unique_ptr<eznve::encoder> encoder;

		std::shared_ptr<concurrencpp::worker_thread_executor> render_queue;
		std::shared_ptr<concurrencpp::thread_pool_executor> work_queue;
		std::shared_ptr<concurrencpp::timer_queue> timer_queue;
		std::optional<concurrencpp::timer> send_timer;

		std::atomic_bool is_closed = false;

		rfkt::flamedb& fdb;

		moodycamel::BlockingReaderWriterCircularBuffer<std::vector<char>> chunks;

		decltype(std::chrono::high_resolution_clock::now()) start;

		send_function send;

		std::int64_t total_frames = 0;
		double loops_per_frame;
		double max_bin_time;
		std::optional<double> target_quality;

		unsigned int fps;
	};
}
/*
void session_render_thread(std::shared_ptr<socket_data> ud, rfkt::cuda::context ctx) {
	ctx.make_current_if_not();
	auto tls = rfkt::cuda::thread_local_stream();

	auto time_since_start = [start = std::chrono::high_resolution_clock::now()]() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1'000'000'000.0;
	};

	auto slept_time = 0.0;
	auto pre_fut_time = 0.0;
	auto wait_time = 0.0;
	auto encode_time = 0.0;
	auto denoise_time = 0.0;
	auto total_bin_time = 0.0;
	std::size_t total_draws = 0;
	std::size_t total_passes = 0;

	auto max_bin_ms = 1000.0 / ud->fps - ud->frame_fudge;
	SPDLOG_INFO("expecting {:.3} ms bin times", max_bin_ms);

	while (!ud->closed) {

		auto t = ud->total_frames * ud->loops_per_frame;

		auto pre_fut_start = time_since_start();

		auto state = ud->kernel.warmup(tls, ud->flame, ud->pp.input_dims(), t, 1, ud->loops_per_frame, 0xdeadbeef, 128);
		auto result_fut = ud->kernel.bin(tls, state, 100000, max_bin_ms, 1'000'000'000);

		auto gamma = ud->flame.gamma.sample(t);
		auto brightness = ud->flame.brightness.sample(t);
		auto vibrancy = ud->flame.vibrancy.sample(t);

		// while we wait, let's encode the last frame
		if (ud->total_frames > 0) {
			auto encode_start = time_since_start();

			auto chunk = ud->session->submit_frame(
				((ud->total_frames - 1) % (ud->fps) == 0) ? eznve::frame_flag::idr: eznve::frame_flag::none
			);

			if (chunk) {
				ud->feed(std::move(chunk->data), ud);
			}

			encode_time += time_since_start() - encode_start;
		}
		pre_fut_time += time_since_start() - pre_fut_start;

		auto wait_start = time_since_start();
		auto result = result_fut.get();
		total_bin_time += result.elapsed_ms;
		total_draws += result.total_draws;
		total_passes += result.total_passes;
		wait_time += time_since_start() - wait_start;

		denoise_time += ud->pp.post_process(state.bins, rfkt::cuda_view<uchar4>{ ud->session->buffer(), ud->session->buffer_size() }, result.quality, gamma, brightness, vibrancy, false, tls).get();

		ud->total_frames++;

		double avg_quality = (total_draws / double(ud->pp.input_dims().x * ud->pp.input_dims().y)) / ud->total_frames;
		double real_fps = ud->total_frames / time_since_start();
		if (ud->total_frames > 0 && ud->total_frames % (ud->fps * 3) == 0) {

			double factor = 1000.0 / ud->total_frames;

			double passes_per_ms = (total_passes / 1'000'000.0) / total_bin_time;
			double draws_per_ms = (total_draws / 1'000'000.0) / total_bin_time;
				
			auto sent_bytes = ud->session->total_bytes();

			SPDLOG_INFO("{:.4} MB, {:.3} mbps, {:.3} avg quality, {:.3}m passes/ms, {:.3}m draws/ms, {:.3} ms/frame avg, {:.3} ms/frame to future get, {:.3} ms/frame future wait. {:.4} real fps, {:.3} ms/frame postprocess avg, {:.3} ms bin average",
				sent_bytes / (1024.0 * 1024.0),
				(8.0 * sent_bytes / (1024.0 * 1024.0)) / (time_since_start()),
				avg_quality,
				passes_per_ms,
				draws_per_ms,
				(time_since_start() - slept_time) * factor,
				pre_fut_time * factor,
				wait_time * factor,
				real_fps,
				denoise_time / ud->total_frames,
				total_bin_time / ud->total_frames);
		}

		//if (result.quality < avg_quality * .8) {
		//	SPDLOG_INFO("Dropped frame detected, {} quality", result.quality);
		//}

		//if (ud->total_frames / double(ud->fps) - time_since_start() > 2 && real_fps > ) {
		if( real_fps > ud->fps * 1.1) {
			SPDLOG_INFO("Sleeping, overbuffered");
			auto sleep = 1000/(ud->fps * 2);
			slept_time += sleep / 1000;
			std::this_thread::sleep_for(std::chrono::milliseconds{ sleep });
		}
	}

	SPDLOG_INFO("Exiting render thread");
}*/

int main(int argc, char** argv) {

	rfkt::flamedb fdb;

	SPDLOG_INFO("Starting refrakt-server");
	try {
		rfkt::initialize(fdb, "config");
	}
	catch(const flang::parse_error& e) {
		SPDLOG_ERROR("Failed to parse variations file: {} ({}:{})", e.what(), e.pos().line, e.pos().column);
		return 1;
	}
	catch (const std::exception& e) {
		SPDLOG_ERROR("Failed to initialize flame system: {}", e.what());
		return 1;
	}


	SPDLOG_INFO("Flame system initialized: {} variations", fdb.variations().size());

	
	auto ctx = rfkt::cuda::init();
	auto dev = ctx.device();

	//rfkt::gpuinfo::init();
	rfkt::denoiser::init(ctx);

	SPDLOG_INFO("Using device {}, CUDA {}.{}", dev.name(), dev.compute_major(), dev.compute_minor());
	
	concurrencpp::runtime runtime;

	auto km = ezrtc::compiler{ std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::cache_adaptors::guarded>(std::make_shared<ezrtc::sqlite_cache>("k2.sqlite3"))) };
	km.find_system_cuda();
	km.add_include_path("assets/kernels/ezrtc_std");
	km.add_include_path("assets/kernels/include");
	auto fc = rfkt::flame_compiler{ km };
	auto tm = rfkt::tonemapper{ km };

	auto jpeg_stream = rfkt::cuda_stream{};

	auto conv = rfkt::converter{ km };
	auto jpeg = rfkt::nvjpeg::encoder{ jpeg_stream };

	auto do_bench = [&](uint2 dims, bool upscale) {
		auto t = rfkt::denoiser::benchmark(dims, upscale, 10, jpeg_stream);
		auto megapixels_per_sec = dims.x * dims.y / (t / 1000) / 1'000'000.0;

		std::cout << std::format("{}\t{}\n", dims.x * dims.y, int(megapixels_per_sec));
	};

	auto jpeg_executor = runtime.make_worker_thread_executor();
	jpeg_executor->post([ctx]() {
		ctx.make_current();
	});

	auto app = uWS::SSLApp();

	uWS::Loop::get()->defer([ctx]() {
		ctx.make_current_if_not();
	});

	auto local_flames = rfkt::fs::list("assets/flames_test", rfkt::fs::filter::has_extension(".flam3"));

	// yeah yeah yeah, rand is bad. not cryptographically securing bank transactions here
	srand(time(0));

	using ws_t = uWS::WebSocket<true, true, rfkt::render_socket::handle>;

	auto ws_open = [&runtime, &ctx, &fc, &km, &fdb](ws_t* ws) {
		auto* handle = ws->getUserData();
		(*handle) = rfkt::render_socket::make(
			fc, km, fdb,
			ctx,
			runtime,
			[ws, loop = uWS::Loop::get()](rfkt::render_socket::handle&& sock, std::vector<char>&& data) {
			loop->defer([ws, sock = std::move(sock), data = std::move(data)]() {
				if (not sock->closed()) {
					ws->send(std::string_view{ data.data(), data.size() });
				}
			});
		});
	};

	auto ws_message = [](ws_t* ws, std::string_view message, uWS::OpCode opCode) {
		(*ws->getUserData())->on_message(message);
	};

	auto ws_closed = [](ws_t* ws, int code, std::string_view msg) {
		(*ws->getUserData())->on_close();
		SPDLOG_INFO("close requested");
	};

	app.ws<rfkt::render_socket::handle>("/stream", {
		.compression = uWS::DISABLED,
		.maxPayloadLength = 10 * 1024 * 1024,
		.idleTimeout = 16,
		.maxBackpressure = 100 * 1024 * 1024,
		.closeOnBackpressureLimit = false,
		.resetIdleTimeoutOnSend = false,
		.sendPingsAutomatically = true,
		.open = ws_open,
		.message = ws_message,
		.drain = nullptr,
		.ping = nullptr,
		.pong = nullptr,
		.close = ws_closed
	});

	app.get("/bananas", [](auto* res, auto* req) {
		auto page = rfkt::fs::read_string("assets/static/muxer.html");

		res->writeStatus("200 OK");
		res->tryEnd(page);
	});

	/*app.get("/render/:id", [&](auto* res, auto* req) {

		SPDLOG_INFO("get {}?{}", req->getUrl(), req->getQuery());
		static const query_parser qp{
			{"width", query_parser::integer},
			{"height", query_parser::integer},
			{"quality", query_parser::decimal},
			{"time", query_parser::decimal},
			{"jpeg_quality", query_parser::integer},
			{"fps", query_parser::integer},
			{"loop_speed", query_parser::integer },
			{"bin_time", query_parser::integer},
			{"segments", query_parser::integer },
			{"noisy", query_parser::boolean},
			{"seed", query_parser::integer},
			{"upscale", query_parser::boolean}
		};

		auto data = qp.parse(req->getQuery());

		struct render_data {
			unsigned int width, height, outwidth, outheight, fps, loop_speed, jpeg_quality, bin_time, segments, seed;
			double time, quality;
			bool noisy, upscale;
		} rd{};

		rd.width = data.value("width", 1920);
		rd.height = data.value("height", 1080);
		rd.quality = data.value("quality", 128.0f);
		rd.time = data.value("time", 0.0f);
		rd.jpeg_quality = data.value("jpeg_quality", 100);
		rd.fps = data.value("fps", 30);
		rd.loop_speed = data.value("loop_speed", 5000);
		rd.bin_time = data.value("bin_time", 2000);
		rd.segments = data.value("segments", 1);
		rd.seed = data.value("seed", 0xdeadbeef);
		rd.noisy = data.value("noisy", false);
		rd.upscale = data.value("upscale", false);

		if (rd.width > 3840) rd.width = 3840;
		if (rd.height > 2160) rd.height = 2160;
		if (rd.quality > 20480) rd.quality = 20480;
		if (rd.bin_time > 20000) rd.bin_time = 20000;
		if (rd.jpeg_quality <= 0 || rd.jpeg_quality > 100) rd.jpeg_quality = 85;

		rd.outwidth = rd.width;
		rd.outheight = rd.height;

		if (rd.upscale && !rd.noisy) {
			rd.width /= 2;
			rd.height /= 2;
		}

		auto cd = http_connection_data::make(res);
		auto flame_path = fmt::format("assets/flames_test/electricsheep.{}.flam3", req->getParameter(0));

		if (!rfkt::fs::exists(flame_path)) {
			end_error<"404", "Not Found">(res, req);
			return;
		}

		jpeg_executor->post([flame_path = std::move(flame_path), cd = std::move(cd), rd = std::move(rd), ctx = ctx, &fc, &tm, &jpeg, &conv, &jpeg_stream, &fdb]() mutable {

			auto timer = rfkt::timer();
			auto fopt = rfkt::import_flam3(fdb, flame_path);
			auto t_load = timer.count();

			if (!fopt) {
				if(!cd->aborted()) end_error<"500", "Internal Server Error">(cd->response());
				return;
			}

			timer.reset();
			auto k_result = [&]() -> rfkt::flame_compiler::result {
				try {
					return fc.get_flame_kernel(fdb, rfkt::precision::f32, fopt.value());
				}
				catch (const std::exception& e) {
					SPDLOG_ERROR("{}", e.what());
					throw;
				}
			}();
			auto t_kernel = timer.count();

			if (!k_result.kernel.has_value()) {
				SPDLOG_INFO("{}", k_result.source);
				SPDLOG_INFO("{}", k_result.log);
				if (!cd->aborted()) end_error<"500", "Internal Server Error>">(cd->response());
				return;
			}

			SPDLOG_INFO("{}", k_result.source);

			auto dn = rfkt::denoiser{ {rd.outwidth, rd.outheight}, rd.upscale };

			auto kernel = rfkt::flame_kernel{ std::move(k_result.kernel.value()) };
			auto tonemapped = rfkt::cuda_buffer<half3>(rd.width * rd.height, jpeg_stream);
			auto smoothed = rfkt::cuda_buffer<half3>(rd.outwidth * rd.outheight, jpeg_stream);
			auto render = rfkt::cuda_buffer<uchar4>(rd.outwidth * rd.outheight, jpeg_stream);
			auto state = kernel.warmup(jpeg_stream, fopt.value(), { rd.width, rd.height }, rd.time, rd.segments, double(rd.fps) / rd.loop_speed, rd.seed, 100);

			timer.reset();
			auto meter = rfkt::gpuinfo::power_meter{ rfkt::gpuinfo::device::by_index(0) };
			auto result = kernel.bin(jpeg_stream, state, rd.quality, rd.bin_time, 1'000'000'000).get();
			auto joules = meter.joules();
			auto t_bin = result.elapsed_ms;

			auto power = joules / (t_bin / 1000.0);
			auto power_per_draw = joules / (result.total_draws / 1'000'000'000.0);

			auto mpass_per_ms = result.total_passes / (1'000'000 * t_bin);
			auto mdraw_per_ms = result.total_draws / (1'000'000 * t_bin);

			SPDLOG_INFO("load {:.4}, kernel {:.4}, bin {:.4}, quality {:.4}, {:.4} mdraw/ms, {:.4} miter/ms, {} passes_per_thread, {:.4} watts, {:.4} joules per billion draws", t_load, t_kernel, t_bin, result.quality, mdraw_per_ms, mpass_per_ms, result.passes_per_thread, power, power_per_draw);

			tm.run(state.bins, tonemapped, { rd.width, rd.height }, result.quality, fopt->gamma.sample(rd.time), fopt->brightness.sample(rd.time), fopt->vibrancy.sample(rd.time), jpeg_stream);
			dn.denoise({ rd.outwidth, rd.outheight }, tonemapped, smoothed, jpeg_stream);
			conv.to_24bit((!rd.noisy)? smoothed: tonemapped, render, { rd.outwidth, rd.outheight }, true, jpeg_stream);
			auto data = jpeg.encode_image(render.ptr(), rd.outwidth, rd.outheight, rd.jpeg_quality, jpeg_stream).get();
 
			std::map<std::string, double> metadata = {};
			metadata["load-ms"] = t_load;
			metadata["kernel-ms"] = t_kernel;
			metadata["bin-ms"] = t_bin;
			metadata["quality"] = result.quality;

			cd->defer([data = std::move(data), meta=std::move(metadata)](auto& cd) {
				if (!cd.aborted()) {
					for (const auto& [k, v] : meta) {
						cd.response()->writeHeader(k, std::format("{}", v));
					}
					cd.response()->writeHeader("Content-Type", "image/jpeg");
					cd.response()->tryEnd({(char*)data.data(), data.size()});
				}
			});
		});

	});*/

	app.listen(3000,
		[](auto* listen_socket) {
			if (listen_socket) {
				SPDLOG_INFO("Listening on port 3000");
			}
		});

	app.run();

	return 1;
}