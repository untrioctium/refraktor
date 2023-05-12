#include <librefrakt/flame_compiler.h>
#include <librefrakt/anima.h>
#include <librefrakt/util/filesystem.h>

#include <gui.h>

namespace rfkt::gui {

	class render_modal {
	public:

		render_modal(ezrtc::compiler& km, rfkt::flame_compiler& fc, const rfkt::flamedb& fdb, rfkt::function_table& ft) : km(km), fc(fc), fdb(fdb), ft(ft) {}

		void do_frame(const rfkt::flame& f, bool open) {
			
			if(open) ImGui::OpenPopup("Render Flame");

			IMFTW_POPUP("Render Flame", true, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove) {
				frame_logic(f);
			}
		}

	private:

		void frame_logic(const rfkt::flame&);
		void launch_worker(const rfkt::flame&);

		ezrtc::compiler& km;
		rfkt::flame_compiler& fc;
		const rfkt::flamedb& fdb;
		rfkt::function_table& ft;

		struct render_params_t {
			int2 dims = { 1280, 720 };
			int fps = 30;
			double seconds_per_loop = 5.0;
			double num_loops = 4.0;
			double target_quality = 512;
			double max_seconds_per_frame = 2;
			fs::path output_file;
		};

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