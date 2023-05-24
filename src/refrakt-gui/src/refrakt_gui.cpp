#include <cuda.h>
#include <concurrencpp/concurrencpp.h>
#include <RtMidi.h>

#include <imgui.h>
#include <imftw/imftw.h>
#include <imftw/gui.h>

#include <librefrakt/flame_info.h>
#include <librefrakt/flame_compiler.h>
#include <librefrakt/image/tonemapper.h>
#include <librefrakt/image/denoiser.h>
#include <librefrakt/image/converter.h>
#include <librefrakt/util/cuda.h>
#include <librefrakt/util/nvjpeg.h>
#include <librefrakt/util/filesystem.h>
#include <librefrakt/util/gpuinfo.h>
#include <librefrakt/util/http.h>

#include "gui/modals/render_modal.h"
#include "gui/panels/preview_panel.h"
#include "gui/panels/flame_editor.h"
#include "gui/panels/timeline.h"
#include "command_executor.h"

#include <IconsMaterialDesign.h>

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


			ImGui::PushStyleColor(ImGuiCol_Text, temp_color);
			ImGui::Text(ICON_MD_THERMOSTAT, temp);
			ImGui::PopStyleColor();
			ImGui::Text("(%d \xc2\xb0" "C)", temp);

			static auto power_sample_pct = dev.wattage_percent();
			static auto power_sample_watts = dev.wattage();
			static auto power_sample_time = imftw::time();

			auto time = imftw::time();
			if (time - power_sample_time > .5) {
				power_sample_pct = dev.wattage_percent();
				power_sample_watts = dev.wattage();
				power_sample_time = time;
			}

			ImGui::Text("%d%% Power (%d Watts)", power_sample_pct, power_sample_watts);

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

bool shortcut_pressed(ImGuiModFlags mods, ImGuiKey key) {
	if (!ImGui::IsKeyPressed(key, false)) return false;

	if (mods & ImGuiModFlags_Alt && !ImGui::IsKeyDown(ImGuiKey_ModAlt)) return false;
	if (mods & ImGuiModFlags_Ctrl && !ImGui::IsKeyDown(ImGuiKey_ModCtrl)) return false;
	if (mods & ImGuiModFlags_Shift && !ImGui::IsKeyDown(ImGuiKey_ModShift)) return false;
	if (mods & ImGuiModFlags_Super && !ImGui::IsKeyDown(ImGuiKey_ModSuper)) return false;
	return true;
}

std::string_view shortcut_to_string(ImGuiModFlags mods, ImGuiKey key) {

	thread_local std::map<std::pair<ImGuiModFlags, ImGuiKey>, std::string> str_cache;
	if(auto it = str_cache.find({ mods, key }); it != str_cache.end()) return it->second;

	std::string result;
	if (mods & ImGuiModFlags_Ctrl) result += "Ctrl+";
	if (mods & ImGuiModFlags_Shift) result += "Shift+";
	if (mods & ImGuiModFlags_Alt) result += "Alt+";
	if (mods & ImGuiModFlags_Super) result += "Super+";
	result += ImGui::GetKeyName(key);
	result = "     " + result;
	
	return str_cache[{ mods, key }] = result;
}

struct menu_item {

	using enabled_func_t = std::move_only_function<bool()>;

	std::string name;
	std::string icon;
	std::optional<ImVec4> icon_color;
	thunk_t action;
	std::optional<std::pair<ImGuiModFlags, ImGuiKey>> shortcut;
	enabled_func_t enabled;

	menu_item& set_name(std::string_view n) { name = n; return *this; }
	menu_item& set_action(thunk_t a) { action = std::move(a); return *this; }
	menu_item& set_enabled(enabled_func_t e) { enabled = std::move(e); return *this; }
	menu_item& set_shortcut(ImGuiModFlags mods, ImGuiKey key) { shortcut = { mods, key }; return *this; }
	menu_item& set_icon(std::string_view i, const std::optional<ImVec4>& color = {}) { icon = i; icon_color = color; return *this; }
};

struct menu_separator {};

class menu_tree {
public:
	using child_type = std::variant<menu_item, menu_tree, menu_separator>;

	std::string name;
	std::list<child_type> items;

	menu_item& add_item(std::string_view name) {
		items.emplace_back(menu_item{ std::string{name} });
		return std::get<menu_item>(items.back());
	}

	menu_tree& add_menu(std::string_view name) {
		items.emplace_back(menu_tree{ std::string{name} });
		return std::get<menu_tree>(items.back());
	}

	void add_separator() {
		items.emplace_back(menu_separator{});
	}

	menu_item* process() {
		menu_item* selected = nullptr;
		if (imftw::menu m{ name, !items.empty() }; m) {
			for (auto& child : items) {
				if (std::holds_alternative<menu_item>(child)) {
					if (auto child_result = process_child_item(&std::get<menu_item>(child)); child_result != nullptr) selected = child_result;
				}
				else if (std::holds_alternative<menu_tree>(child)) {
					if (auto child_result = std::get<menu_tree>(child).process(); child_result != nullptr) selected = child_result;
				}
				else if (std::holds_alternative<menu_separator>(child)) {
					ImGui::Separator();
				}
			}
		}

		if (selected == nullptr) {
			auto shortcut = process_shortcuts();
			if (shortcut != nullptr) selected = shortcut;
		}

		return selected;
	}

private:

	menu_item* process_shortcuts() {
		menu_item* selected = nullptr;
		for (auto& child : items) {
			if (std::holds_alternative<menu_item>(child)) {
				auto& item = std::get<menu_item>(child);
				if (item.shortcut && shortcut_pressed(item.shortcut->first, item.shortcut->second)) {
					selected = &item;
					break;
				}
			}
			else if (std::holds_alternative<menu_tree>(child)) {
				if (auto child_result = std::get<menu_tree>(child).process_shortcuts(); child_result != nullptr) selected = child_result;
			}
		}
		return selected;
	}

	menu_item* process_child_item(menu_item* item) {

		const auto space_width = ImGui::CalcTextSize(" ").x;
		const auto max_icon_width = ImGui::GetFontSize();
		const auto num_spaces = (max_icon_width / space_width) + 1;

		bool enabled = !item->enabled || item->enabled();
		menu_item* selected = nullptr;
		auto cursor_before = ImGui::GetCursorScreenPos();
		const char* shortcut = (item->shortcut)? shortcut_to_string(item->shortcut->first, item->shortcut->second).data() : nullptr;  
		if (ImGui::MenuItem(std::format("{}{}     ", std::string(num_spaces, ' '), item->name).c_str(), shortcut, nullptr, enabled) && item->action) {
			selected = item;
		}
		
		if (!item->icon.empty()) {
			auto icon_color = enabled ? (item->icon_color ? *item->icon_color : ImGui::GetStyleColorVec4(ImGuiCol_Text)) : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
			auto advance = (max_icon_width - ImGui::CalcTextSize(item->icon.c_str()).x) / 2;
			cursor_before.x += advance;
			ImGui::GetForegroundDrawList()->AddText(cursor_before, ImGui::GetColorU32(icon_color), item->icon.c_str());
		}
		return selected;
	}
};

class main_menu {
public:
	menu_tree& add_menu(std::string_view name) {
		return menus.emplace_back(std::string{ name });
	}

	menu_item* process() {
		auto menu_scope = imftw::main_menu{};
		if (!menu_scope) return nullptr;

		menu_item* selected = nullptr;
		for (auto& menu : menus) {
			if(auto result = menu.process(); result != nullptr) selected = result; 
		}
		return selected;
	}

private:
	std::list<menu_tree> menus;
};

namespace midi {
	
	struct message {
		enum class type_t {
			none,
			note_on,
			note_off,
			control_change,
			pitch_bend,
			program_change,
			aftertouch,
			poly_aftertouch
		} type = type_t::none;

		uint8_t channel = 0;
		std::pair<uint8_t, uint8_t> data;

		static message from_bytes(std::span<uint8_t> msg) noexcept {
			message m{};

			if (msg.size() == 0)
				return m;

			m.channel = msg[0] & 0x0F;

			switch (msg[0] & 0xF0) {
			case 0x80:
				m.type = type_t::note_off;
				m.data = { msg[1], msg[2] };
				break;
			case 0x90:
				m.type = type_t::note_on;
				m.data = { msg[1], msg[2] };
				break;
			case 0xA0:
				m.type = type_t::poly_aftertouch;
				m.data = { msg[1], msg[2] };
				break;
			case 0xB0:
				m.type = type_t::control_change;
				m.data = { msg[1], msg[2] };
				break;
			case 0xC0:
				m.type = type_t::program_change;
				m.data = { msg[1], 0 };
				break;
			case 0xD0:
				m.type = type_t::aftertouch;
				m.data = { msg[1], 0 };
				break;
			case 0xE0:
				m.type = type_t::pitch_bend;
				m.data = { msg[1], msg[2] };
				break;
			}

			return m;
		}
	};

	namespace detail {
		RtMidiIn& info_instance() {
			static RtMidiIn instance{};
			return instance;
		}
	};

	auto device_count() {
		return detail::info_instance().getPortCount();
	}

	std::string device_name(unsigned int port) {
		return detail::info_instance().getPortName(port);
	}

}

class midi_system {
public:

private:
	std::map<unsigned int, RtMidiIn> inputs;
};


template<typename Func>
auto show_splash(std::string_view task_name, Func&& f) {
	imftw::begin_frame();
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->Pos);
	ImGui::SetNextWindowSize(viewport->Size);

	IMFTW_WINDOW("Loading...", ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings) {
		auto window_size = ImGui::GetWindowSize();
		auto text_size = ImGui::CalcTextSize(task_name.data());
		ImGui::SetCursorPos({ (window_size.x - text_size.x) / 2, window_size.y - text_size.y * 2 });
		ImGui::TextUnformatted(task_name.data());
	}

	if constexpr (std::same_as<void, decltype(f())>) {
		f();
		imftw::end_frame();
		return;
	}
	else {
		auto result = f();
		imftw::end_frame();
		return result;
	}
}

void about_window() {

	const static std::map<std::string_view, std::tuple<std::string_view, std::string_view, std::string_view>> libraries {
		{ "spdlog", { "Gabi Melman", "github.com/gabime/spdlog", "MIT" }},
		{ "zlib-ng", {"Jean-loup Gailly and Mark Adler", "github.com/zlib-ng/zlib-ng", "zlib"} },
		{ "rtmidi", {"Gary P. Scavone", "github.com/thestk/rtmidi", "MIT"} },
		{ "implot", {"Evan Pezent", "github.com/epezent/implot", "MIT"} },
		{ "reproc", {"Daan De Meyer", "github.com/DaanDeMeyer/reproc", "MIT"} },
		{ "iconfontcppheaders", {"Juliette Foucaut and Doug Binks", "github.com/juliettef/IconFontCppHeaders", "zlib"} },
		{ "glfw", {"Marcus Geelnard and Camilla Lowy", "github.com/glfw/glfw", "zlib"} },
		{ "glad", {"David Herberth", "github.com/Dav1dde/glad", "MIT"} },
		{ "freetype", {"David Turner, Robert Wilhelm, Werner Lemberg", "freetype.org", "FTL"} },
		{ "imgui", {"Omar Cornut", "github.com/ocornut/imgui", "MIT"} },
		{ "concurrentqueue", {"Cameron Desrochers", "github.com/cameron314/concurrentqueue", "BSD"} },
		{ "readerwriterqueue", {"Cameron Desrochers", "github.com/cameron314/readerwriterqueue", "BSD"} },
		{ "dylib", {"Martin Olivier", "github.com/martin-olivier/dylib", "MIT"} },
		{ "pegtl", {"Daniel Frey and Dr. Colin Hirsch", "github.com/taocpp/PEGTL", "Boost"} },
		{ "yaml-cpp", {"Jesse Beder", "github.com/jbeder/yaml-cpp", "MIT"} },
		{ "xxhash", {"Yann Collet", "github.com/Cyan4973", "BSD 2-Clause"} },
		{ "sqlite", {"D. Richard Hipp", "www.sqlite.org/index.html", "Public Domain"} },
		{ "concurrencpp", {"David Haim", "github.com/David-Haim/concurrencpp", "MIT"} },
		{ "stb", {"Sean Barrett", "github.com/nothings/stb", "MIT"} },
		{ "cppcodec", {"Jakob Petsovits and Topology Inc.", "github.com/tplgy/cppcodec", "MIT"} },
		{ "inja", {"Berscheid", "github.com/pantor/inja", "MIT"} },
		{ "json", {"Niels Lohmann", "github.com/nlohmann/json", "MIT"} },
		{ "pugixml", {"Arseny Kapoulkine", "github.com/zeux/pugixml", "MIT"} },
		{ "lua", {"Lua.org, PUC-Rio. ", "lua.org", "MIT"} },
		{ "sol3", {"Rapptz, ThePhD, and contributors", "github.com/ThePhD/sol2", "MIT"} },
		{ "curl", {"Daniel Stenberg", "github.com/curl/curl", "MIT"} },
		{ "zip", {"Kuba Podgorski", "github.com/kuba--/zip", "Unlicense"} }
	};

	const float max_library_name = ImGui::CalcTextSize(std::max_element(libraries.begin(), libraries.end(), [](const auto& a, const auto& b) {
		return a.first.size() < b.first.size();
		})->first.data()).x + 5.0;

	const float max_author = ImGui::CalcTextSize(std::get<0>(std::max_element(libraries.begin(), libraries.end(), [](const auto& a, const auto& b) {
		return std::get<0>(a.second).size() < std::get<0>(b.second).size();
		})->second).data()).x + 5.0;

	const float max_url = ImGui::CalcTextSize(std::get<1>(std::max_element(libraries.begin(), libraries.end(), [](const auto& a, const auto& b) {
		return std::get<1>(a.second).size() < std::get<1>(b.second).size();
		})->second).data()).x + 5.0;

	const float max_license = ImGui::CalcTextSize(std::get<2>(std::max_element(libraries.begin(), libraries.end(), [](const auto& a, const auto& b) {
		return std::get<2>(a.second).size() < std::get<2>(b.second).size();
		})->second).data()).x + 5.0;

	IMFTW_POPUP("About", true, ImGuiWindowFlags_AlwaysAutoResize) {
		imftw::text_centered("Refrakt v0.1.0 by Alex Riley");
		ImGui::PushStyleColor(ImGuiCol_Text, 0xFFB59550);
		imftw::text_centered("github.com/untrioctium/refraktor");
		auto min = ImGui::GetItemRectMin();
		auto max = ImGui::GetItemRectMax();
		min.y = max.y;
		ImGui::GetWindowDrawList()->AddLine(min, max, 0xFFB59550, 1.0f);
		ImGui::PopStyleColor();

		if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
		if (ImGui::IsItemClicked(0)) imftw::open_browser("https://github.com/untrioctium/refraktor");

		imftw::text_centered("Refrakt is distributed under the MIT License");

		ImGui::TextUnformatted("");
		imftw::text_centered("Based on the original flam3 renderer by Scott Draves");
		imftw::text_centered("Uses CUDA, NVENC, and Optix, which are property of NVIDIA");
		imftw::text_centered("Distributed with ffmpeg.exe on Windows, which is property of the FFmpeg Project");
		imftw::text_centered("Material Design Icons provided by Google under the Apache 2.0 license");
		ImGui::TextUnformatted("");
		imftw::text_centered("Refrakt uses the following C/C++ libraries. Thank you to all of the authors and countless contributors!");
		ImGui::BeginChild("##about_libs", { max_library_name + max_author + max_url + max_license + 20, 400 }, true);
		ImGui::BeginTable("##libraries", 4);
		ImGui::TableSetupColumn("Name", 0, max_library_name);
		ImGui::TableSetupColumn("Author(s)", 0, max_author);
		ImGui::TableSetupColumn("URL", 0, max_url);
		ImGui::TableSetupColumn("License", 0, max_license);

		ImGui::TableHeadersRow();

		for (const auto& [name, info] : libraries) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(name.data());
			ImGui::TableNextColumn();
			ImGui::TextUnformatted(std::get<0>(info).data());
			ImGui::TableNextColumn();
			ImGui::PushStyleColor(ImGuiCol_Text, 0xFFB59550);
			ImGui::TextUnformatted(std::get<1>(info).data());
			ImGui::PopStyleColor();

			auto min = ImGui::GetItemRectMin();
			auto max = ImGui::GetItemRectMax();
			min.y = max.y;
			ImGui::GetWindowDrawList()->AddLine(min, max, 0xFFB59550, 1.0f);

			if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			if (ImGui::IsItemClicked(0)) imftw::open_browser(std::format("https://{}", std::get<1>(info)));

			ImGui::TableNextColumn();
			ImGui::TextUnformatted(std::get<2>(info).data());
		}

		ImGui::EndTable();
		ImGui::EndChild();

		if(ImGui::Button("Close") || ImGui::IsKeyReleased(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
	}
}

struct flame_sequencer : public rfkt::gui::panel::timeline::interface {

	std::pair<flame_track, flame_track> flame_tracks;

	int item_count() const override { return 2; }
	int item_type_count() const override { return 1; }
	std::string_view item_type_name(int type) const override { return "flame"; }
	std::string_view item_name(int id) const override { return (id == 0) ? "Flame Track 1" : "Flame Track 2"; };

	int item_type(int id) const override { return 0; }
	int item_segment_count(int id) const override {
		if (id == 0) return flame_tracks.first.segment_count();
		else return flame_tracks.second.segment_count();
	}

	time_span& item_segment(int item_id, int seg_id) override {
		if (item_id == 0) return flame_tracks.first[seg_id].span;
		else return flame_tracks.second[seg_id].span;
	}

	time_span::int_t min_frame() const override { return 0; }
	time_span::int_t max_frame() const override { return 3600 * 60 * 5; }

};

namespace rfkt {
	class project {
	public:

		using id_t = std::size_t;

		id_t add_flame(const flame& f) {
			auto id = id_counter++;
			flames_[id] = f;
			return id;
		}

		bool remove_flame(id_t id) {
			return flames_.erase(id) > 0;
		}

		bool has_flame(id_t id) const {
			return flames_.contains(id);
		}

		flame& get_flame(id_t id) {
			return flames_[id];
		}

		auto flames(this auto&& self) {
			return std::views::all(self.flames_);
		}

	private:

		id_t id_counter = 0;

		std::map<std::size_t, flame> flames_;

		flame_sequencer sequencer;
	};
}

class refrakt_app {
public:

	refrakt_app() = default;

	void run() {
		if (!initialize()) return;

		imftw::sig::set_window_decorated(true).get();
		imftw::sig::set_window_maximized(true).get();
		imftw::sig::set_vsync_enabled(false);
		imftw::sig::set_target_framerate(144);

		while (true) {
			imftw::begin_frame();

			ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

			if (auto selected = mainm.process(); selected) selected->action();

			IMFTW_WINDOW("\xee\xa2\xb8" "Flame Options") {
				rfkt::gui::panels::flame_editor(fdb, cur_flame, c_exec, functions);
			}

			ImGui::ShowDemoWindow();
			draw_status_bar();
			about_window();

			//ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
			//IMFTW_WINDOW("Sequencer") {
				//ImSequencer::Sequencer(&seq, &current_frame, &expanded, &selected_entry, &first_frame, seq_options);
			//	rfkt::gui::panel::timeline::show(&seq, current_frame, 1);
			//}
			//ImGui::PopStyleVar();

			prev_panel->show(fdb, cur_flame, functions);
			render_modal->do_frame(cur_flame);

			imftw::end_frame();
		}
	}

private:

	bool initialize() {

		auto monitor_size = imftw::sig::get_monitor_size().get();
		auto monitor_pos = imftw::sig::get_monitor_position().get();
		imftw::sig::set_window_decorated(false).get();
		imftw::sig::set_window_size(1280, 720).get();
		imftw::sig::set_window_position(monitor_pos.first + (monitor_size.first - 1280) / 2, monitor_pos.second + (monitor_size.second - 720) / 2).get();
		imftw::sig::set_window_visible(true).get();
		imftw::sig::set_vsync_enabled(true);

		ctx = show_splash("Initializing CUDA", []() { return rfkt::cuda::init(); });
		show_splash("Initializing denoising system", [&]() { rfkt::denoiser::init(ctx); });
		show_splash("Initializing GPU statistics system", []() { rfkt::gpuinfo::init(); });
		show_splash("Loading variations", [&]() {rfkt::initialize(fdb, "config"); });

		auto runtime_path = show_splash("Setting up CUDA runtime", []() { return rfkt::cuda::check_and_download_cudart(); });
		if (!runtime_path) {
			SPDLOG_CRITICAL("Could not find CUDA runtime");
			return false;
		}


		k_comp = show_splash("Setting up compiler", [&runtime_path]() {
			auto kernel_cache = std::make_shared<ezrtc::sqlite_cache>((rfkt::fs::user_local_directory() / "kernel.sqlite3").string().c_str());
			auto zlib = std::make_shared<ezrtc::cache_adaptors::zlib>(std::make_shared<ezrtc::cache_adaptors::guarded>(kernel_cache));
			auto compiler = std::make_shared<ezrtc::compiler>( zlib );
			compiler->find_system_cuda(*runtime_path);
			return compiler;
		});

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

		f_comp = show_splash("Setting up flame compiler", [&]() { return std::make_shared<rfkt::flame_compiler>(k_comp); });
		tonemap = show_splash("Creating tonemapper", [&] { return std::make_shared<rfkt::tonemapper>( *k_comp ); });
		denoise_normal = show_splash("Creating denoiser", [] { return std::make_shared<rfkt::denoiser>( uint2{2560, 1440}, false ); });
		denoise_upscale = show_splash("Creating upscaling denoiser", [] { return std::make_shared<rfkt::denoiser>(uint2{ 2560, 1440 }, true); });
		convert = show_splash("Creating converter", [&] { return std::make_shared<rfkt::converter>(*k_comp); });

		preview_panel::renderer_t renderer = [&](
			rfkt::cuda_stream& stream, const rfkt::flame_kernel& kernel,
			rfkt::flame_kernel::saved_state& state,
			rfkt::flame_kernel::bailout_args bo, double3 gbv, bool upscale) {

				const auto total_bins = state.bin_dims.x * state.bin_dims.y;

				const auto output_dims = (upscale) ? uint2{ state.bin_dims.x * 2, state.bin_dims.y * 2 } : state.bin_dims;
				const auto output_bins = output_dims.x * output_dims.y;

				auto tonemapped = rfkt::cuda_buffer<half3>{ total_bins, stream };
				auto denoised = rfkt::cuda_buffer<half3>{ output_bins, stream };
				auto out_buf = rfkt::cuda_buffer<uchar4>{ output_bins, stream };

				auto bin_info = kernel.bin(stream, state, bo).get();
				state.quality += bin_info.quality;

				tonemap->run(state.bins, tonemapped, state.bin_dims, state.quality, gbv.x, gbv.y, gbv.z, stream);
				auto& denoiser = upscale ? denoise_upscale : denoise_normal;
				denoiser->denoise(output_dims, tonemapped, denoised, stream);
				tonemapped.free_async(stream);
				convert->to_24bit(denoised, out_buf, output_dims, false, stream);
				denoised.free_async(stream);
				stream.sync();

				return out_buf;
		};

		auto render_worker = runtime.make_worker_thread_executor();
		render_worker->post([&]() {ctx.make_current(); });
		preview_panel::executor_t executor = [render_worker = std::move(render_worker)](std::move_only_function<void(void)>&& func) {
			render_worker->post(std::move(func));
		};

		prev_panel = std::make_unique<preview_panel>(*f_comp, std::move(executor), std::move(renderer), c_exec);
		render_modal = std::make_unique<rfkt::gui::render_modal>(*k_comp, *f_comp, fdb, functions );

		cur_flame = std::move(rfkt::import_flam3(fdb, rfkt::fs::read_string("assets/flames_test/electricsheep.247.47670.flam3")).value());

		auto& file_menu = mainm.add_menu("File");
		file_menu.add_item("New project");
		file_menu.add_item("Open project...");
		file_menu.add_item("Save project");
		file_menu.add_item("Save project as...");

		file_menu.add_separator();

		file_menu.add_item("Exit").set_shortcut(ImGuiModFlags_Alt, ImGuiKey_F4);

		auto& edit_menu = mainm.add_menu("Edit");
		edit_menu.add_item("Undo")
			.set_icon(ICON_MD_UNDO, ImVec4(1.0, 0.0f, 0.0f, 1.0f))
			.set_shortcut(ImGuiModFlags_Ctrl, ImGuiKey_Z)
			.set_action([&]() { c_exec.undo(); })
			.set_enabled([&]() { return c_exec.can_undo(); });

		edit_menu.add_item("Redo")
			.set_shortcut(ImGuiModFlags_Ctrl, ImGuiKey_Y)
			.set_icon(ICON_MD_REDO, ImVec4(0.0, 1.0f, 0.0f, 1.0f))
			.set_action([&]() { c_exec.redo(); })
			.set_enabled([&]() { return c_exec.can_redo(); });

		edit_menu.add_separator();
		edit_menu.add_item("Copy source")
			.set_action([&]() { imftw::sig::set_clipboard(f_comp->make_source(fdb, cur_flame)); })
			.set_icon(ICON_MD_DATA_OBJECT);

		auto& project_menu = mainm.add_menu("Project");
		project_menu
			.add_item("Import flame...")
			.set_icon(ICON_MD_LOCAL_FIRE_DEPARTMENT)
			.set_shortcut(ImGuiModFlags_Ctrl, ImGuiKey_O)
			.set_action([&]() {
				runtime.background_executor()->enqueue([&]() mutable {
					ctx.make_current_if_not();

					std::string filename = imftw::show_open_dialog(rfkt::fs::working_directory() / "assets" / "flames", "Flames\0 * .flam3\0");

					if (filename.empty()) return;

					auto flame = rfkt::import_flam3(fdb, rfkt::fs::read_string(filename));
					if (!flame) {
						SPDLOG_ERROR("could not open flame: {}", filename);
						return;
					}

					SPDLOG_INFO("flame data:\n{}", flame->serialize().dump(2));

					rfkt::hash_t hash;
					rfkt::timer t;
					for(int i = 0; i < 10'000; ++i)
						hash ^= flame->value_hash() | hash;

					auto time = t.count() / 10'000.0 * 1'000.0 * 1'000.0;
					SPDLOG_INFO("hash: {} ({} us per hash)", hash.str64(), time);

					imftw::defer([&, flame = std::move(flame.value())]() mutable {
						c_exec.clear();
						cur_flame = std::move(flame);
					});
				});
			});

		auto& render_menu = mainm.add_menu("Render");
		render_menu.add_item("Video")
			.set_icon(ICON_MD_VIDEO_FILE)
			.set_action([&]() {
				render_modal->trigger_open();
			});

		auto& options_menu = mainm.add_menu("Options");
		auto& style_menu = options_menu.add_menu("Style");

		rfkt::gui::set_style(1);

		for (const auto& style : rfkt::gui::get_styles()) {
			auto& item = style_menu.add_item(style.name).set_action([id = style.id]() { rfkt::gui::set_style(id); });
			if (style.name == "Decay 1998") {
				item.set_icon("\xce\xbb", ImVec4(251.0 / 255, 126.0 / 255, 20.0 / 255, 1.0f));
			}
		}

		auto& help_menu = mainm.add_menu("Help");
		help_menu.add_item("About")
			.set_icon(ICON_MD_INFO)
			.set_action([&]() { ImGui::OpenPopup("About"); });

		return true;
	}

	command_executor c_exec;
	rfkt::flamedb fdb;
	rfkt::cuda::context ctx;

	std::shared_ptr<ezrtc::compiler> k_comp;
	std::shared_ptr<rfkt::flame_compiler> f_comp;
	rfkt::function_table functions;

	std::shared_ptr<rfkt::tonemapper> tonemap;
	std::shared_ptr<rfkt::denoiser> denoise_normal;
	std::shared_ptr<rfkt::denoiser> denoise_upscale;
	std::shared_ptr<rfkt::converter> convert;

	concurrencpp::runtime runtime;

	std::unique_ptr<preview_panel> prev_panel;
	std::unique_ptr<rfkt::gui::render_modal> render_modal;

	rfkt::flame cur_flame;

	main_menu mainm;
};

int main(int argc, char**) {

	auto app = refrakt_app{};

	imftw::run("Refrakt", [&]() { app.run(); });
	return 0;
}