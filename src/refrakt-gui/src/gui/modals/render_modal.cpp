#include <imftw/imftw.h>
#include <imftw/gui.h>

#include <reproc++/reproc.hpp>

#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/converter.h>
#include <librefrakt/image/tonemapper.h>

#include <librefrakt/util/stb.h>

#include <eznve.hpp>

#include "gui/modals/render_modal.h"
#include "gl.h"

#include <iostream>

std::string make_mux_command(const rfkt::fs::path& in, const rfkt::fs::path& out, int fps) {
	return std::format(R"(""{}/bin/ffmpeg.exe" -y -i "{}" -c copy "{}")", rfkt::fs::working_directory().string().c_str(), in.string().c_str(), out.string().c_str());
}

void rfkt::gui::render_modal::frame_logic(const rfkt::flame& flame) {

	bool can_start = true;

	IMFTW_WITH_ENABLED(!worker_state, 0.5) {
		ImGui::InputInt2("Dimensions", (int*)&render_params.dims);
		can_start &= render_params.dims.x > 0 && render_params.dims.y > 0;

		ImGui::InputInt("FPS", &render_params.fps);
		can_start &= render_params.fps > 0;

		ImGui::InputDouble("Seconds per loop", &render_params.seconds_per_loop);
		can_start &= render_params.seconds_per_loop > 0;

		ImGui::InputDouble("Number of loops", &render_params.num_loops);
		can_start &= render_params.num_loops > 0;

		ImGui::InputDouble("Target quality", &render_params.target_quality);
		can_start &= render_params.target_quality > 0;

		ImGui::InputDouble("Max seconds per frame", &render_params.max_seconds_per_frame);
		can_start &= render_params.max_seconds_per_frame > 0;

		if (ImGui::Button("Browse")) {
			auto selected = ImFtw::ShowSaveDialog(rfkt::fs::user_home_directory(), "MP4 Video\0*.mp4\0");
			if (!selected.empty()) {
				render_params.output_file = selected;
			}
		}
		ImGui::SameLine();

		can_start &= !render_params.output_file.empty();
		if (render_params.output_file.empty()) {
			ImGui::TextColored(ImVec4(1.0f, .2f, .2f, 1.0f), "No output file selected");
		}
		else
			ImGui::Text("%s", render_params.output_file.string().c_str());
	}

	const auto total_seconds = render_params.num_loops * render_params.seconds_per_loop;
	const auto total_frames = static_cast<int>(total_seconds * render_params.fps);

	ImGui::Text("%f seconds, %d frames", total_seconds, total_frames);

	if (worker_state) {
		ImGui::SameLine();

		auto done_frames = worker_state->rendered_frames->load();
		double total_real_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_time).count() / 1e9;
		double real_frame_rate = done_frames / total_real_seconds;

		int estimated_seconds_left = (total_frames - done_frames) / real_frame_rate;
		
		int minutes_left = estimated_seconds_left / 60;
		int seconds_left = estimated_seconds_left % 60;

		ImGui::Text("%.2f %d:%02d", real_frame_rate / render_params.fps, minutes_left, seconds_left);

		if (ImGui::Button("Cancel")) {
			worker_state->worker.request_stop();
			worker_state->worker.join();
			worker_state.reset();
			ImFtw::Sig::SetWindowProgressMode(ImFtw::Sig::ProgressMode::Disabled);
		}
		else if (worker_state->done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			worker_state->worker.join();
			worker_state.reset();
			ImFtw::Sig::SetWindowProgressMode(ImFtw::Sig::ProgressMode::Disabled);
		}
		else {
			ImGui::SameLine();
			if (done_frames == total_frames) {
				ImGui::ProgressBar(1.0f, ImVec2(-FLT_MIN, 0.0f), "Muxing...");
			}
			else ImGui::ProgressBar(done_frames / float(total_frames));

			ImFtw::Sig::SetWindowProgressValue(done_frames, total_frames);
		}
	}

	else {

		IMFTW_WITH_ENABLED(can_start, 0.5) {
			if (ImGui::Button("Start")) {
				launch_worker(flame);
				start_time = std::chrono::steady_clock::now();
				ImFtw::Sig::SetWindowProgressMode(ImFtw::Sig::ProgressMode::Determinate);
				ImFtw::Sig::SetWindowProgressValue(0, total_frames);
				
			}
		}

		ImGui::SameLine();
		if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
	}
}

namespace rfkt {
	class postprocessor {
	public:
		postprocessor(ezrtc::compiler& kc, uint2 dims, rfkt::denoiser_flag::flags dn_opts = rfkt::denoiser_flag::none) :
			tm(kc),
			dn(dims, dn_opts),
			conv(kc),
			tonemapped(dn_opts& rfkt::denoiser_flag::upscale ? dims.x / 2 : dims.x, dn_opts& rfkt::denoiser_flag::upscale ? dims.y / 2 : dims.y),
			denoised(dims.x, dims.y),
			dims_(dims),
			upscale(upscale)
		{

		}

		~postprocessor() = default;

		postprocessor(const postprocessor&) = delete;
		postprocessor& operator=(const postprocessor&) = delete;

		postprocessor(postprocessor&&) = default;
		postprocessor& operator=(postprocessor&&) = default;

		auto make_output_buffer() const -> roccu::gpu_buffer<uchar4> {
			return roccu::gpu_buffer<uchar4>{ dims_.x* dims_.y };
		}

		auto make_output_buffer(RUstream stream) const -> roccu::gpu_buffer<uchar4> {
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
			roccu::gpu_span<uchar4> out,
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
			dn.denoise(tonemapped, denoised, stream);
			conv.to_32bit(denoised, out, planar_output, stream);
			stream.host_func([&t = perf_timer, p = std::move(promise)]() mutable {
				p.set_value(t.count());
				});

			return future;
		}

	private:
		rfkt::tonemapper tm;
		rfkt::denoiser_old dn;
		rfkt::converter conv;

		rfkt::gpu_image<half3> tonemapped;
		rfkt::gpu_image<half3> denoised;

		rfkt::timer perf_timer;

		uint2 dims_;
		bool upscale;
	};
}

struct last_frame_t {
	roccu::gpu_buffer<float4> bins = {};
	double quality;
	std::array<double, 3> gbv;
};

void rfkt::gui::render_modal::launch_worker(const rfkt::flame& flame)
{
	auto done_promise = std::promise<bool>();
	auto done_future = done_promise.get_future();

	auto frame_counter = std::make_shared<std::atomic_uint32_t>(0);

	auto worker_thread = 
	[
		promise = std::move(done_promise),
		render_params = this->render_params,
		&fc = this->fc,
		&km = this->km,
		&fdb = this->fdb,
		&ft = this->ft,
		flame = flame, // intentional copy
		frame_counter,
		ctx = roccu::context::current()
	]
	(std::stop_token stoke) mutable {

		ctx.make_current();

		auto dims = uint2{
			static_cast<unsigned int>(render_params.dims.x),
			static_cast<unsigned int>(render_params.dims.y)
		};

		auto pp = rfkt::postprocessor{ km, dims, rfkt::denoiser_flag::none };
		auto post_stream = roccu::gpu_stream{};
		auto encoder = eznve::encoder{ dims, {static_cast<unsigned int>(render_params.fps), 1}, eznve::codec::hevc, ctx };

		//auto chunkfile_name = std::format("{}.h265", render_params.output_file.string());
		//auto chunkfile = std::ofstream{ chunkfile_name, std::ios::binary };

		auto ffmpeg_options = reproc::options{};
		auto ffmpeg_process = reproc::process{};

		ffmpeg_process.start(std::vector<std::string>{
			std::format("{}\\bin\\ffmpeg.exe", rfkt::fs::working_directory().string()),
				/*"-hide_banner", "-loglevel", "error", */"-y", "-i", "-", "-c", "copy", "-r", std::to_string(render_params.fps), render_params.output_file.string()
		});


		auto kernel_result = fc.get_flame_kernel(fdb, rfkt::precision::f32, flame);
		if (!kernel_result.kernel) {
			promise.set_value(false);
			return;
		}

		const auto& kernel = kernel_result.kernel.value();
		
		const auto total_seconds = render_params.num_loops * render_params.seconds_per_loop;
		const auto total_frames = static_cast<int>(total_seconds * render_params.fps);

		auto com_queue = com_queue_t{};
		auto bin_worker = std::jthread(&binning_thread, ctx, total_frames, std::cref(flame), std::cref(render_params), std::ref(ft), std::cref(kernel), std::ref(com_queue));

		int chunks_processed = 0;

		auto& i = *frame_counter;
		while(i != total_frames) {
			if (stoke.stop_requested()) {
				bin_worker.request_stop();
				bin_worker.join();

				//chunkfile.close();
				ffmpeg_process.kill();
				//std::filesystem::remove(chunkfile_name);
				promise.set_value(false);
				return;
			}

			auto binfo = bin_info{};
			if (!com_queue.wait_dequeue_timed(binfo, std::chrono::microseconds(100))) {
				continue;
			}

			auto encoder_input = roccu::gpu_span<uchar4>{ encoder.buffer(), encoder.buffer_size() };
			pp.post_process(binfo.bins, encoder_input, binfo.quality, binfo.gbv.x, binfo.gbv.y, binfo.gbv.z, false, post_stream).get();
			auto chunks = encoder.submit_frame((i % render_params.fps == 0 || i + 1 == total_frames) ? eznve::frame_flag::idr : eznve::frame_flag::none);

			for (auto& chunk : chunks) {
				ffmpeg_process.write((uint8_t*)chunk.data.data(), chunk.data.size());//chunkfile.write(chunk.data.data(), chunk.data.size());
				chunks_processed++;
			}
			i++;
		}

		
		for (auto chunks = encoder.flush(); auto& chunk : chunks) {
			ffmpeg_process.write((uint8_t*) chunk.data.data(), chunk.data.size());
			chunks_processed++;
		}

		SPDLOG_INFO("processed {} chunks", chunks_processed);

		ffmpeg_process.close(reproc::stream::in);
		ffmpeg_process.wait(reproc::milliseconds(1000));

		//chunkfile.close();
		//int res = std::system(make_mux_command(chunkfile_name, render_params.output_file.string().c_str(), render_params.fps).c_str());
		//std::filesystem::remove(chunkfile_name);

		promise.set_value(true);
	};

	worker_state = {
		std::jthread{std::move(worker_thread)},
		std::move(frame_counter),
		std::move(done_future)
	};
}

void rfkt::gui::render_modal::binning_thread(std::stop_token stoke, roccu::context ctx, int total_frames, const rfkt::flame& f, const render_params_t& rp, rfkt::function_table& ft, const rfkt::flame_kernel& kernel, com_queue_t& queue)
{
	ctx.make_current();

	const auto loops_per_frame = 1.0 / (rp.fps * rp.seconds_per_loop);
	auto invoker = ft.make_invoker();
	auto stream = roccu::gpu_stream{};
	auto bailout = rfkt::flame_kernel::bailout_args{ .millis = static_cast<std::uint32_t>(rp.max_seconds_per_frame * 1000), .quality = rp.target_quality };

	const auto dims = uint2{
		static_cast<unsigned int>(rp.dims.x),
		static_cast<unsigned int>(rp.dims.y)
	};


	for (int i = 0; i < total_frames && !stoke.stop_requested(); i++) {
		const auto t = loops_per_frame * i;

		std::vector<double> samples = {};
		auto packer = [&samples](double v) { samples.push_back(v); };
		f.pack_sample(packer, invoker, t - 1.2 * loops_per_frame, dims.x, dims.y);
		f.pack_sample(packer, invoker, t, dims.x, dims.y);
		f.pack_sample(packer, invoker, t + 1.2 * loops_per_frame, dims.x, dims.y);
		f.pack_sample(packer, invoker, t + 2.4 * loops_per_frame, dims.x, dims.y);

		auto state = kernel.warmup(stream, samples, dims, 0xdeadbeef, 100);
		auto bin_result = kernel.bin(stream, state, bailout).get();

		auto miters_per_ms = (bin_result.total_passes / 1'000'000.0) / bin_result.elapsed_ms;
		auto mdraws_per_ms = (bin_result.total_draws / 1'000'000.0) / bin_result.elapsed_ms;

		SPDLOG_INFO("{} miters/ms, {} mdraws/ms", miters_per_ms, mdraws_per_ms);

		while(queue.size_approx() > 5) { 
			if (stoke.stop_requested()) return;
			SPDLOG_INFO("Downstream buffers full");
			std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
		}

		queue.emplace(std::move(state.bins), bin_result.quality,
			double3{
				f.gamma.sample(t, invoker),
				f.brightness.sample(t, invoker),
				f.vibrancy.sample(t, invoker)
			});
	}
}
