#include <cuda.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <concurrencpp/concurrencpp.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/image/tonemapper.h>
#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/converter.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/nvjpeg.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/gpuinfo.h>

#include "gl.h"
#include "gui.h"

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

void draw_status_bar() {
	ImGuiViewportP* viewport = (ImGuiViewportP*)(void*)ImGui::GetMainViewport();
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	if (ImGui::BeginViewportSideBar("##StatusBar", viewport, ImGuiDir_Down, ImGui::GetFrameHeight(), window_flags)) {
		if (ImGui::BeginMenuBar()) {
			auto dev = rfkt::gpuinfo::device::by_index(0);
			auto temp = dev.temperature();

			auto temp_color = [&]() {

				constexpr auto coolest = 30.0;
				constexpr auto hottest = 80.0;

				if (temp <= coolest) return IM_COL32(0, 255, 0, 255);
				if (temp >= hottest) return IM_COL32(255, 0, 0, 255);

				auto mix = (temp - coolest) / (hottest - coolest);
				return IM_COL32(255 * mix, 255 * (1 - mix), 0, 255);
			}();


			ImGui::Text("GPU:");
			ImGui::PushStyleColor(ImGuiCol_Text, temp_color);
			ImGui::Text("%d C", temp);
			ImGui::PopStyleColor();
			ImGui::Text("%d W", dev.wattage());

			ImGui::EndMenuBar();
		}
		ImGui::End();
	}
	ImGui::PopStyleVar();

	/*if (ImGui::BeginViewportSideBar("##ToolBar", viewport, ImGuiDir_Up, ImGui::GetFrameHeight(), window_flags)) {
		if (ImGui::BeginMenuBar()) {
			ImGui::EndMenuBar();
		}
		ImGui::End();
	}*/
}

bool flame_editor(rfkt::flamedb& fdb, rfkt::flame& f) {
	bool modified = false;

	rfkt::gui::id_scope flame_scope{ &f };

	ImGui::Text("Display");
	modified |= rfkt::gui::drag_double("Rotate", f.rotate, 0.01, -360, 360);
	modified |= rfkt::gui::drag_double("Scale", f.scale, 0.1, 0, 10'000);
	modified |= rfkt::gui::drag_double("Center X", f.center_x, 0.1, -10, 10);
	modified |= rfkt::gui::drag_double("Center Y", f.center_y, 0.1, -10, 10);
	ImGui::Separator();

	ImGui::Text("Color");
	modified |= rfkt::gui::drag_double("Gamma", f.gamma, 0.01, 0, 5);
	modified |= rfkt::gui::drag_double("Brightness", f.brightness, 0.01, 0, 100);
	modified |= rfkt::gui::drag_double("Vibrancy", f.vibrancy, 0.1, 0, 100);
	ImGui::Separator();

	f.for_each_xform([&](int xid, rfkt::xform& xf) {
		rfkt::gui::id_scope xf_scope{ &xf };

		std::string xf_name = (xid == -1) ? "Final Xform" : fmt::format("Xform {}", xid);

		if (ImGui::CollapsingHeader(xf_name.c_str())) {
			modified |= rfkt::gui::drag_double("Weight", xf.weight, 0.01, 0, 50);
			modified |= rfkt::gui::drag_double("Color", xf.color, 0.001, 0, 1);
			modified |= rfkt::gui::drag_double("Color Speed", xf.color_speed, 0.001, 0, 1);
			modified |= rfkt::gui::drag_double("Opacity", xf.opacity, 0.001, 0, 1);

			if (ImGui::BeginTabBar("Vchain")) {
				for (int i = 0; i < xf.vchain.size(); i++) {
					if (ImGui::BeginTabItem(std::format("{}", i).c_str())) {
						ImGui::BeginChild(std::format("vchain{}", i).c_str(), ImVec2(0, 200));

						auto& vl = xf.vchain[i];
						{
							rfkt::gui::id_scope vl_scope{ &vl };

							modified |= rfkt::gui::drag_double("Affine A", vl.transform.a, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine B", vl.transform.b, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine C", vl.transform.c, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine D", vl.transform.d, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine E", vl.transform.e, 0.001, -5, 5);
							modified |= rfkt::gui::drag_double("Affine F", vl.transform.f, 0.001, -5, 5);
							ImGui::Separator();
							static bool just_opened = false;
							static char filter[128] = { 0 };

							if (ImGui::Button("Add variation")) {
								ImGui::OpenPopup("add_variation");
								just_opened = true;
								std::memset(filter, 0, sizeof(filter));
							}

							if (ImGui::BeginPopup("add_variation")) {
								if (just_opened) {
									just_opened = false;
									ImGui::SetKeyboardFocusHere(-1);
								}

								ImGui::InputText("##filter", filter, sizeof(filter));
								std::string_view selected_var{};

								if (ImGui::BeginListBox("##vlist", ImVec2(-FLT_MIN, 10 * ImGui::GetTextLineHeightWithSpacing()))) {
									for (const auto& var : fdb.variations()) {
										if (!vl.has_variation(var.name) && (strlen(filter) < 2 || var.name.find(filter) != std::string::npos)) {
											if (ImGui::Selectable(var.name.c_str())) {
												selected_var = var.name;
											}
										}
									}
									ImGui::EndListBox();
								}
								ImGui::EndPopup();
							}

							for (auto& [vname, vd] : vl) {
								ImGui::Separator();
								modified |= rfkt::gui::drag_double(vname.c_str(), vd.weight, 0.001, -50, 50);

								for (auto& [pname, val] : vd) {
									modified |= rfkt::gui::drag_double(std::format("{}_{}", vname, pname).c_str(), val, 0.001, -50, 50);
								}
							}
						}
						ImGui::EndChild();
						ImGui::EndTabItem();
					}
				}
				ImGui::EndTabBar();
			}
		}
	});

	return modified;
}

auto enqueue_render(
	std::shared_ptr<concurrencpp::worker_thread_executor> executor,
	rfkt::cuda_stream& stream,
	std::shared_ptr<rfkt::flame_kernel> kernel,
	std::shared_ptr<rfkt::flame_kernel::saved_state> state, 
	rfkt::flame_kernel::bailout_args bo, 
	rfkt::tonemapper& tm,
	rfkt::denoiser& dn,
	rfkt::converter& conv,
	double3 gbv) {

	auto out_tex = rfkt::gl::texture{ state->bin_dims.x, state->bin_dims.y };
	auto cuda_map = out_tex.map_to_cuda();

	return executor->submit([&stream, state, bo, kernel = std::move(kernel), out_tex = std::move(out_tex), &tm, &dn, &conv, gbv, cuda_map = std::move(cuda_map)]() mutable {
		const auto total_bins = state->bin_dims.x * state->bin_dims.y;

		auto out_buf = rfkt::cuda_buffer<uchar4>{ total_bins, stream };
		auto tonemapped = rfkt::cuda_buffer<half3>{ total_bins, stream };
		auto denoised = rfkt::cuda_buffer<half3>{ total_bins, stream };

		auto bin_info = kernel->bin(stream, *state, bo).get();
		state->quality += bin_info.quality;
		
		tm.run(state->bins, tonemapped, state->bin_dims, state->quality, gbv.x, gbv.y, gbv.z, stream);
		dn.denoise(state->bin_dims, tonemapped, denoised, stream);
		tonemapped.free_async(stream);
		conv.to_24bit(denoised, out_buf, state->bin_dims, false, stream);
		denoised.free_async(stream);
		stream.sync();

		cuda_map.copy_from(out_buf);

		out_buf.free_async(stream);

		return std::move(out_tex);
	});
}

class render_panel {

};

void main_thread(rfkt::cuda::context ctx, std::stop_source stop) {

	namespace ccpp = concurrencpp;

	ctx.make_current();

	rfkt::flamedb fdb;
	rfkt::initialize(fdb, "config");

	auto kernel_cache = std::make_shared<ezrtc::sqlite_cache>("kernel.sqlite3");
	auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::cache_adaptors::guarded>(kernel_cache));

	ezrtc::compiler kc{ zlib };
	kc.find_system_cuda();

	rfkt::flame_compiler fc(kc);
	auto tm = rfkt::tonemapper{ kc };
	auto dn = rfkt::denoiser{ {2560, 1440}, false };
	auto conv = rfkt::converter{ kc };

	std::optional<rfkt::gl::texture> displayed_tex;
	std::optional<ccpp::result<rfkt::gl::texture>> rendering;
	std::shared_ptr<rfkt::flame_kernel::saved_state> current_state;

	auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string("assets/flames_test/electricsheep.247.47670.flam3"));
	auto compile_result = fc.get_flame_kernel(fdb, rfkt::precision::f32, flame.value());

	auto kernel = std::make_shared<rfkt::flame_kernel>(std::move(compile_result.kernel.value()));

	rfkt::gl::make_current();
	rfkt::gl::set_target_fps(120);
	bool should_exit = false;

	ccpp::runtime runtime;

	auto render_worker = runtime.make_worker_thread_executor();
	auto render_stream = rfkt::cuda_stream{};
	render_worker->post([ctx]() {ctx.make_current(); });

	const auto seconds_per_loop = 10000000000000.0;
	const auto fps = 30;

	bool trigger_render = false;
	bool next_frame_is_dirty = false;
	bool clear_state = false;
	bool clear_state_next = false;

	bool first_frame = true;

	auto thunk_executor = runtime.make_manual_executor();
	thunk_executor->post([&]() {ctx.make_current(); });

	while (!should_exit) {
		rfkt::gl::begin_frame();
		if (rfkt::gl::close_requested()) should_exit = true;

		while (!thunk_executor->empty()) {
			thunk_executor->loop_once();
		}

		ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

		if (rendering.has_value() && rendering->status() == ccpp::result_status::value) {
			displayed_tex = rendering->get();
			rendering = std::nullopt;
			if (next_frame_is_dirty) {
				next_frame_is_dirty = false;
				trigger_render = true;
				if (clear_state_next) {
					clear_state = true;
					clear_state_next = false;
				}
			}
		}

		if (current_state && !rendering && current_state->quality < 2000) {
			trigger_render = true;
			clear_state |= false;
		}

		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(32 / 255.0f, 32 / 255.0f, 32 / 255.0f, 1.0f));
		if (ImGui::BeginMainMenuBar()) {
			if (ImGui::BeginMenu("File")) {
				ImGui::MenuItem("New");

				if (ImGui::MenuItem("Open...")) {
					runtime.background_executor()->enqueue([ctx, &fdb, &fc, &thunk_executor, &cur_flame = flame, &cur_kernel = kernel, &trigger_render, &clear_state]() mutable {
						ctx.make_current_if_not();

						std::string filename = rfkt::gl::show_open_dialog("Flames\0*.flam3\0");

						if (filename.empty()) return;

						auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string(filename));
						if (!flame) {
							SPDLOG_ERROR("could not open flame: {}", filename);
						}

						auto result = fc.get_flame_kernel(fdb, rfkt::precision::f32, flame.value());

						if (!result.kernel.has_value()) {
							SPDLOG_ERROR("Could not compile {}:\n{}\n{}", result.source, result.log);
							return;
						}

						thunk_executor->enqueue([flame = std::move(flame), kernel = std::move(result.kernel.value()), &cur_flame, &cur_kernel, &trigger_render, &clear_state]() mutable {
							cur_flame = std::move(flame);
							cur_kernel = std::make_shared<rfkt::flame_kernel>(std::move(kernel));
							trigger_render = true;
							clear_state = true;
						});
					});
				}

				ImGui::EndMenu();
			}

			/*if (ImGui::BeginMenu("Style")) {
				auto styles = rfkt::gui::get_styles();

				for (const auto& style : styles) {
					if (ImGui::MenuItem(style.name.c_str())) {
						rfkt::gui::set_style(style.id);
					}
				}
				ImGui::EndMenu();
			}*/

			if (ImGui::BeginMenu("Debug")) {
				if (ImGui::MenuItem("Copy flame source")) {
					rfkt::gl::set_clipboard(fc.make_source(fdb, flame.value()));
				}
				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}
		ImGui::PopStyleVar();
		ImGui::PopStyleColor();

		if (ImGui::Begin("Flame Options")) {
			if (flame_editor(fdb, flame.value())) {
				trigger_render = true;
				clear_state = true;
			}
		}
		ImGui::End();

		ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(.1 * 255, .1 * 255, .1 * 255, 255));
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
		std::optional<uint2> render_size{};
		if (ImGui::Begin("Render") && !first_frame) {
			auto window_min = ImGui::GetWindowContentRegionMin();
			auto window_max = ImGui::GetWindowContentRegionMax();

			auto avail = ImGui::GetContentRegionAvail();

			render_size = uint2{ (uint32_t)(window_max.x - window_min.x), (uint32_t)(window_max.y - window_min.y) };
			auto lol = 2;
		}
		ImGui::End();
		ImGui::PopStyleColor();
		ImGui::PopStyleVar();

		if (!render_size) {
			trigger_render = true;
			clear_state = true;
		}


		ImGui::ShowDemoWindow();

		draw_status_bar();

		if (render_size.has_value() && current_state && 
			(render_size.value().x != current_state->bin_dims.x || render_size.value().y != current_state->bin_dims.y)) {
			trigger_render = true;
			clear_state = true;
		}

		static bool dragging = false;
		static auto last_delta = ImGui::GetMouseDragDelta();
		bool preview_hovered = false;
		if (ImGui::Begin("Render")) {
			if (displayed_tex.has_value()) {
				auto& tex = displayed_tex.value();
				ImGui::Image((void*)(intptr_t)tex.id(), ImVec2(render_size->x, render_size->y));
				preview_hovered = ImGui::IsItemHovered();
			}
		}
		ImGui::End();

		if (!dragging && preview_hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
			dragging = true;
			last_delta = ImGui::GetMouseDragDelta();
		}
		else if (dragging) {
			if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) dragging = false;
			else {
				auto delta = ImGui::GetMouseDragDelta();
				auto drag_dist = delta - last_delta;
				if (drag_dist.x != 0 || drag_dist.y != 0) {
					auto scale = flame->scale;
					auto rot = -flame->rotate * IM_PI / 180;

					double2 vector = { drag_dist.x / (scale * render_size->y) , drag_dist.y / (scale * render_size->y) };
					double2 rotated = { vector.x * cos(rot) - vector.y * sin(rot), vector.x * sin(rot) + vector.y * cos(rot) };

					flame->center_x -= rotated.x;
					flame->center_y -= rotated.y;

					trigger_render = true;
					clear_state = true;
				}

				last_delta = delta;
			}
		}

		if (preview_hovered) {
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		}

		if (auto scroll = ImGui::GetIO().MouseWheel; scroll != 0 && preview_hovered) {
			flame->scale *= std::pow(1.1, scroll);
			trigger_render = true;
			clear_state = true;
		}

		bool is_rendering = rendering.has_value();

		if (trigger_render && is_rendering) {
			trigger_render = false;
			next_frame_is_dirty = true;
			clear_state_next |= clear_state;
			clear_state = false;
		}

		if (!is_rendering && trigger_render && render_size) {
			if (!current_state || clear_state) {
				auto samples = gen_samples(flame.value(), 0.0, 1 / (fps * seconds_per_loop));
				auto new_state = kernel->warmup(render_stream, samples, render_size.value(), 0xdeadbeef, 100);
				current_state = std::make_shared<rfkt::flame_kernel::saved_state>(std::move(new_state));
				clear_state = false;
			}
			rendering = enqueue_render(
				render_worker, render_stream, kernel, current_state,
				{ .millis = 1000/60, .quality = 2000 }, tm, dn, conv,
				{ flame->gamma, flame->brightness, flame->vibrancy }
			);
			trigger_render = false;
		}

		rfkt::gl::end_frame(true);
		first_frame = false;

	}

	if (rendering.has_value()) {
		auto _ = rendering->get();
	}

	stop.request_stop();
	glfwPostEmptyEvent();
}

int main(int argc, char**) {

	auto ctx = rfkt::cuda::init();
	rfkt::denoiser::init(ctx);
	rfkt::gpuinfo::init();

	if (!rfkt::gl::init(1920, 1080)) {
		return 1;
	}
	rfkt::gui::set_style(3);

	std::stop_source stopper;
	auto stoke = stopper.get_token();

	auto render_thread = std::jthread(main_thread, ctx, std::move(stopper));

	rfkt::gl::event_loop(stoke);

	return 0;
}