#include <imftw/imftw.h>
#include <imftw/gui.h>

#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/converter.h>
#include <librefrakt/image/tonemapper.h>

#include <librefrakt/util/stb.h>

#include <eznve.hpp>

#include "gui/modals/render_modal.h"
#include "gl.h"

std::string make_mux_command(const rfkt::fs::path& in, const rfkt::fs::path& out, int fps) {
	return std::format(R"(""{}/bin/mp4mux.exe" --track h264:"{}"#frame_rate={} "{}"")", rfkt::fs::working_directory().string().c_str(), in.string().c_str(), fps, out.string().c_str());
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
			auto selected = imftw::show_save_dialog(rfkt::fs::user_home_directory(), "MP4 Video\0*.mp4\0");
			if (!selected.empty()) {
				render_params.output_file = selected;
			}
		}
		ImGui::SameLine();

		can_start &= !render_params.output_file.empty();
		if (render_params.output_file.empty()) {
			ImGui::TextColored(ImVec4(1, .2, .2, 1), "No output file selected");
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
			imftw::sig::set_window_progress_mode(imftw::sig::progress_mode::disabled);
		}
		else if (worker_state->done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			worker_state->worker.join();
			worker_state.reset();
			imftw::sig::set_window_progress_mode(imftw::sig::progress_mode::disabled);
		}
		else {
			ImGui::SameLine();
			if (done_frames == total_frames) {
				ImGui::ProgressBar(1.0f, ImVec2(-FLT_MIN, 0.0f), "Muxing...");
			}
			else ImGui::ProgressBar(done_frames / float(total_frames));

			imftw::sig::set_window_progress_value(done_frames, total_frames);
		}
	}

	else {

		IMFTW_WITH_ENABLED(can_start, 0.5) {
			if (ImGui::Button("Start")) {
				launch_worker(flame);
				start_time = std::chrono::steady_clock::now();
				imftw::sig::set_window_progress_mode(imftw::sig::progress_mode::determinate);
				imftw::sig::set_window_progress_value(0, total_frames);
				
			}
		}

		ImGui::SameLine();
		if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
	}
}

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
			upscale(upscale) {}

		~postprocessor() = default;

		postprocessor(const postprocessor&) = delete;
		postprocessor& operator=(const postprocessor&) = delete;

		postprocessor(postprocessor&&) = default;
		postprocessor& operator=(postprocessor&&) = default;

		auto make_output_buffer() const -> rfkt::cuda_buffer<uchar4> {
			return rfkt::cuda_buffer<uchar4>{ dims_.x* dims_.y };
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
			rfkt::cuda_span<float4> in,
			rfkt::cuda_span<uchar4> out,
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

struct last_frame_t {
	rfkt::cuda_buffer<float4> bins = {};
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
		ctx = rfkt::cuda::context::current()
	]
	(std::stop_token stoke) mutable {

		ctx.make_current();

		auto stream = rfkt::cuda_stream{};

		auto dims = uint2{
			static_cast<unsigned int>(render_params.dims.x),
			static_cast<unsigned int>(render_params.dims.y)
		};

		auto pp = rfkt::postprocessor{ km, dims, false };
		auto post_stream = rfkt::cuda_stream{};
		auto encoder = eznve::encoder{ dims, {static_cast<unsigned int>(render_params.fps), 1}, eznve::codec::h264, ctx };
		auto encoder_input = rfkt::cuda_span<uchar4>{ encoder.buffer(), encoder.buffer_size() / 4 };

		auto chunkfile_name = std::format("{}.h264", render_params.output_file.string());
		auto chunkfile = std::ofstream{ chunkfile_name, std::ios::binary };

		auto kernel_result = fc.get_flame_kernel(fdb, rfkt::precision::f32, flame);
		if (!kernel_result.kernel) {
			promise.set_value(false);
			return;
		}

		const auto& kernel = kernel_result.kernel.value();
		
		const auto total_seconds = render_params.num_loops * render_params.seconds_per_loop;
		const auto total_frames = static_cast<int>(total_seconds * render_params.fps);
		const auto loops_per_frame = 1.0 / (render_params.fps * render_params.seconds_per_loop);

		auto invoker = [&ft]<typename... Args>(Args&&... args) { return ft.call(std::forward<Args>(args)...); };

		std::optional<last_frame_t> lf;


		auto& i = *frame_counter;
		while(i != total_frames) {
			if (stoke.stop_requested()) {
				chunkfile.close();
				std::filesystem::remove(chunkfile_name);
				promise.set_value(false);
				return;
			}

			const auto t = loops_per_frame * i;

			auto post_future = [&]() -> std::optional<std::future<double>> {
				if (lf) {
					return pp.post_process(lf->bins, encoder_input, lf->quality, lf->gbv[0], lf->gbv[1], lf->gbv[2], false, post_stream);
				}
				else return std::nullopt;
			}();

			std::vector<double> samples = {};
			auto packer = [&samples](double v) { samples.push_back(v); };
			flame.pack_sample(packer, invoker, t - 1.2 * loops_per_frame, dims.x, dims.y);
			flame.pack_sample(packer, invoker, t, dims.x, dims.y);
			flame.pack_sample(packer, invoker, t + 1.2 * loops_per_frame, dims.x, dims.y);
			flame.pack_sample(packer, invoker, t + 2.4 * loops_per_frame, dims.x, dims.y);


			auto state = kernel.warmup(stream, samples, dims, 0xdeadbeef, 100);
			auto bailout = rfkt::flame_kernel::bailout_args{ .millis = static_cast<std::uint32_t>(render_params.max_seconds_per_frame * 1000), .quality = render_params.target_quality };
			auto bin_future = kernel.bin(stream, state, bailout);

			if (lf) {
				post_future->get();
				auto chunk = encoder.submit_frame((i % render_params.fps == 0) ? eznve::frame_flag::idr : eznve::frame_flag::none);

				if (chunk) {
					chunkfile.write(chunk->data.data(), chunk->data.size());
				}
				i++;

			}
			
			auto bin_result = bin_future.get();

			if (!lf) {
				lf = last_frame_t{};
			}

			lf->bins = std::move(state.bins);
			lf->quality = bin_result.quality;
			lf->gbv = {
				flame.gamma.sample(t, invoker),
				flame.brightness.sample(t, invoker),
				flame.vibrancy.sample(t, invoker)
			};
		}

		chunkfile.close();
		int res = std::system(make_mux_command(chunkfile_name, render_params.output_file.string().c_str(), render_params.fps).c_str());
		std::filesystem::remove(chunkfile_name);

		promise.set_value(res == 0);
	};

	worker_state = {
		std::jthread{std::move(worker_thread)},
		std::move(frame_counter),
		std::move(done_future)
	};
}
