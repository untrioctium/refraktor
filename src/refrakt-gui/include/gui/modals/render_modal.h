#include <librefrakt/flame_compiler.h>
#include <librefrakt/anima.h>
#include <librefrakt/util/filesystem.h>

#include <gui.h>

#include <readerwriterqueue.h>

namespace rfkt::gui {

	class render_modal {
	public:

		render_modal(ezrtc::compiler& km, rfkt::flame_compiler& fc, const rfkt::flamedb& fdb, rfkt::function_table& ft) : km(km), fc(fc), fdb(fdb), ft(ft) {}

		void trigger_open() { should_open = true; }

		void do_frame(const rfkt::flame& f) {
			
			if (should_open) {
				ImGui::OpenPopup("Render Flame");
				should_open = false;
			}

			IMFTW_POPUP("Render Flame", true, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove) {
				frame_logic(f);
			}
		}

	private:

		void frame_logic(const rfkt::flame&);
		void launch_worker(const rfkt::flame&);

		struct bin_info : public rfkt::traits::noncopyable {
			using bin_storage = decltype(rfkt::flame_kernel::saved_state::bins);
			bin_storage bins;
			double quality;
			double3 gbv;

			bin_info() = default;
			bin_info(bin_storage&& bins, double quality, double3 gbv) noexcept: 
				bins(std::move(bins)), quality(quality), gbv(gbv) {}

			bin_info(bin_info&& o) noexcept {
				*this = std::move(o);
			}

			bin_info& operator=(bin_info&& o) noexcept {
				std::swap(bins, o.bins);
				std::swap(quality, o.quality);
				std::swap(gbv, o.gbv);
				return *this;
			}
		};

		struct render_params_t {
			int2 dims = { 1280, 720 };
			int fps = 30;
			double seconds_per_loop = 5.0;
			double num_loops = 1.0;
			double target_quality = 16;
			double max_seconds_per_frame = 2;
			fs::path output_file;
		};

		using com_queue_t = moodycamel::BlockingReaderWriterQueue<bin_info>;
		static void binning_thread(std::stop_token stoke, roccu::context ctx, int total_frames, const rfkt::flame& f, const render_params_t& rp, rfkt::function_table& ft, const rfkt::flame_kernel& kernel, com_queue_t& queue);

		bool should_open = false;

		ezrtc::compiler& km;
		rfkt::flame_compiler& fc;
		const rfkt::flamedb& fdb;
		rfkt::function_table& ft;

		render_params_t render_params;
		std::chrono::steady_clock::time_point start_time;

		struct worker_state_t {
			std::jthread worker;
			std::shared_ptr<std::atomic_uint32_t> rendered_frames;
			std::future<bool> done;
		};

		std::optional<worker_state_t> worker_state;
	};

}