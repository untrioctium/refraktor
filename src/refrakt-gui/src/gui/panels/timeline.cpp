#include <imgui.h>
#include <imgui_internal.h>

#include "gui/panels/timeline.h"

namespace rfkt::gui::panel::timeline {

#ifndef IMGUI_DEFINE_MATH_OPERATORS
	static ImVec2 operator+(const ImVec2& a, const ImVec2& b) noexcept {
		return ImVec2(a.x + b.x, a.y + b.y);
	}

	static ImVec2 operator-(const ImVec2& a, const ImVec2& b) noexcept {
		return ImVec2(a.x - b.x, a.y - b.y);
	}

	static ImVec2 operator*(const ImVec2& a, float b) noexcept {
		return ImVec2(a.x * b, a.y * b);
	}

	static ImVec2 operator*(float a, const ImVec2& b) noexcept {
		return ImVec2(a * b.x, a * b.y);
	}

	static ImVec2 operator/(const ImVec2& a, float b) noexcept {
		return ImVec2(a.x / b, a.y / b);
	}

#endif

	auto get_color(ImGuiCol color) {
		return ImGui::GetColorU32(ImGui::GetStyle().Colors[color]);
	}

	struct drawing_helper {
		ImDrawList* draw_list;
		const ImVec2 canvas_pos;
		const ImVec2 canvas_size;

		ImVec2 to_screen(const ImVec2& p) const noexcept {
			return canvas_pos + p;
		}

		ImRect to_screen(const ImRect& r) const noexcept {
			return ImRect(canvas_pos + r.Min, canvas_pos + r.Max);
		}

		ImVec2 to_local(const ImVec2& p) const noexcept {
			return p - canvas_pos;
		}

		ImRect to_local(const ImRect& r) const noexcept {
			return ImRect(r.Min - canvas_pos, r.Max - canvas_pos);
		}

		void add_rect(const ImRect& r, ImU32 color, float rounding = 0) const noexcept {
			draw_list->AddRect(to_screen(r.Min), to_screen(r.Max), color, rounding);
		}
		void add_rect_filled(const ImRect& r, ImU32 color, float rounding = 0) const noexcept {
			draw_list->AddRectFilled(to_screen(r.Min), to_screen(r.Max), color, rounding);
		}
		void add_line(const ImVec2& a, const ImVec2& b, ImU32 color, float thickness = 1.0f) const noexcept {
			draw_list->AddLine(to_screen(a), to_screen(b), color, thickness);
		}
		void add_circle(const ImVec2& center, float radius, ImU32 color, int num_segments = 12, float thickness = 1.0f) const noexcept {
			draw_list->AddCircle(to_screen(center), radius, color, num_segments, thickness);
		}
		void add_circle_filled(const ImVec2& center, float radius, ImU32 color, int num_segments = 12) const noexcept {
			draw_list->AddCircleFilled(to_screen(center), radius, color, num_segments);
		}
		void add_text(const ImVec2& pos, ImU32 color, const char* text_begin, const char* text_end = nullptr) const noexcept {
			draw_list->AddText(to_screen(pos), color, text_begin, text_end);
		}
	};

	struct segment_result {
		bool changed;
		time_span::range frame_span;
	};

	struct segment_args {
		// the total width of the segment in frames
		// these may be outside the visible area
		time_span::range frame_span;

		// the maximum bounds of the segment in frames
		time_span::range bounds;

		float mouse_delta_x;

		ImRect scrollbar_rect;
		ImVec2 mouse_pos;

		ImU32 scrollbar_bg;
	};

	auto segment_logic(const drawing_helper& draw, time_span& segment, std::string_view seg_name, std::string_view id_name, const ImRect& seg_rect, time_span::int_t frame_movement_size, ImGuiID dragging_id, double& drag_buildup) {

		const auto handle_size = ImVec2(10, seg_rect.Max.y - seg_rect.Min.y);
		ImGui::SetCursorPos(seg_rect.Min);
		ImGui::InvisibleButton(std::format("{}_left", id_name).c_str(), handle_size);
		auto segment_left_hovered = ImGui::IsItemHovered();
		auto segment_left_active = ImGui::IsItemActive();
		auto segment_left_dragging = dragging_id == ImGui::GetItemID();

		ImGui::SetCursorPos({ seg_rect.Max.x - handle_size.x, seg_rect.Min.y });
		ImGui::InvisibleButton(std::format("{}_right", id_name).c_str(), handle_size);
		auto segment_right_hovered = ImGui::IsItemHovered();
		auto segment_right_active = ImGui::IsItemActive();
		auto segment_right_dragging = dragging_id == ImGui::GetItemID();

		ImGui::SetCursorPos(seg_rect.Min + ImVec2(handle_size.x, 0));
		ImGui::InvisibleButton(std::format("{}_middle", id_name).c_str(), seg_rect.GetSize() - ImVec2(2 * handle_size.x, 0));
		auto segment_middle_hovered = ImGui::IsItemHovered();
		auto segment_middle_active = ImGui::IsItemActive();
		auto segment_middle_dragging = dragging_id == ImGui::GetItemID();

		if (segment_left_dragging && drag_buildup != 0) {
			if (drag_buildup < 0) {
				auto snap_remainder = segment.start() % frame_movement_size;
				int full_movements = std::floor((-drag_buildup - snap_remainder) / frame_movement_size);
				auto total_movement = full_movements * frame_movement_size + snap_remainder;

				if (total_movement && total_movement <= -drag_buildup) {
					segment.adjust_start(-total_movement);
					drag_buildup += total_movement;
				}
			}
			else {
				auto snap_remainder = frame_movement_size - segment.start() % frame_movement_size;
				int full_movements = std::floor((drag_buildup - snap_remainder) / frame_movement_size);
				auto total_movement = full_movements * frame_movement_size + snap_remainder;
				if (total_movement && total_movement <= drag_buildup) {
					segment.adjust_start(total_movement);
					drag_buildup -= total_movement;
				}
			}
		}

		if (segment_right_dragging && drag_buildup != 0) {
			if (drag_buildup < 0) {
				auto snap_remainder = segment.end() % frame_movement_size;
				int full_movements = std::floor((-drag_buildup - snap_remainder) / frame_movement_size);
				auto total_movement = full_movements * frame_movement_size + snap_remainder;
				if (total_movement && total_movement <= -drag_buildup) {
					segment.adjust_end(-total_movement);
					drag_buildup += total_movement;
				}
			}
			else {
				auto snap_remainder = frame_movement_size - segment.end() % frame_movement_size;
				int full_movements = std::floor((drag_buildup - snap_remainder) / frame_movement_size);
				auto total_movement = full_movements * frame_movement_size + snap_remainder;
				if (total_movement <= drag_buildup) {
					segment.adjust_end(total_movement);
					drag_buildup -= total_movement;
				}
			}
		}

		if (segment_middle_dragging && drag_buildup != 0) {
			if (drag_buildup < 0) {
				auto snap_remainder = segment.start() % frame_movement_size;
				int full_movements = std::floor((-drag_buildup - snap_remainder) / frame_movement_size);
				auto total_movement = full_movements * frame_movement_size + snap_remainder;
				if (total_movement && total_movement <= -drag_buildup) {

					drag_buildup -= segment.adjust_end(segment.adjust_start(-total_movement));
				}
			}
			else {
				auto snap_remainder = frame_movement_size - segment.end() % frame_movement_size;
				int full_movements = std::floor((drag_buildup - snap_remainder) / frame_movement_size);
				auto total_movement = full_movements * frame_movement_size + snap_remainder;
				if (total_movement && total_movement <= drag_buildup) {
					drag_buildup -= segment.adjust_start(segment.adjust_end(total_movement));
				}
			}
		}

		auto segment_color = get_color(
			segment_middle_active ? ImGuiCol_ButtonActive :
			segment_middle_hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button
		);

		draw.add_rect_filled(seg_rect, segment_color);

		if (!seg_name.empty()) {
			auto text_size = ImGui::CalcTextSize(seg_name.data());
			if (text_size.x < seg_rect.GetWidth() && text_size.y < seg_rect.GetHeight()) {
				auto text_pos = seg_rect.Min + ImVec2((seg_rect.GetWidth() - text_size.x) / 2, (seg_rect.GetHeight() - text_size.y) / 2);
				draw.add_text(text_pos, get_color(ImGuiCol_Text), seg_name.data());
			}
			else if (segment_middle_hovered) {
				ImGui::BeginTooltip();
				ImGui::TextUnformatted(seg_name.data());
				ImGui::EndTooltip();
			}
		}

		if (segment_left_dragging || segment_left_hovered || segment_right_dragging || segment_right_hovered)
			ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

	}

	auto scrollbar_logic(ImGuiIO& io, interface* iseq, const drawing_helper& draw, const ImVec2& mpos, const ImRect& scrollbar_area, time_span::int_t min_advance) {

		auto scrollbar_width = scrollbar_area.Max.x - scrollbar_area.Min.x;
		auto scrollbar_height = scrollbar_area.Max.y - scrollbar_area.Min.y;

		bool mouse_in_scrollbar = (io.MousePos.x - draw.canvas_pos.x) >= scrollbar_area.Min.x && (io.MousePos.x - draw.canvas_pos.x) <= scrollbar_area.Max.x;

		const auto mouse_delta_frames = (time_span::int_t)(io.MouseDelta.x / scrollbar_width * iseq->max_frame());

		draw.add_rect_filled(scrollbar_area, get_color(ImGuiCol_ScrollbarBg), scrollbar_area.Max.y - scrollbar_area.Min.y);

		static time_span::range frame_span = { 0, std::min(10000, iseq->max_frame()) };

		float left_handle_pct = (float)frame_span.start / iseq->max_frame();
		float right_handle_pct = (float)frame_span.end / iseq->max_frame();

		auto handle_size = ImVec2(10, scrollbar_height - 2);
		auto left_handle_pos = ImVec2(scrollbar_area.Min.x + scrollbar_width * left_handle_pct, scrollbar_area.Min.y + 2);
		auto right_handle_pos = ImVec2(scrollbar_area.Min.x + (scrollbar_width - handle_size.x) * right_handle_pct, scrollbar_area.Min.y + 2);

		auto left_handle_rect = ImRect(left_handle_pos, left_handle_pos + handle_size);
		auto right_handle_rect = ImRect(right_handle_pos, right_handle_pos + handle_size);

		ImGui::SetCursorPos(left_handle_pos + ImVec2(0, handle_size.y));
		ImGui::InvisibleButton("left_handle", handle_size);
		bool left_handle_hovered = ImGui::IsItemHovered();
		bool left_handle_active = ImGui::IsItemActive();

		if (left_handle_active && mouse_in_scrollbar) {
			auto old_start = frame_span.start;
			frame_span.start = std::clamp(old_start + mouse_delta_frames, 0, frame_span.end);
		}

		ImGui::SetCursorPos(right_handle_pos + ImVec2(0, handle_size.y));
		ImGui::InvisibleButton("right_handle", handle_size);
		bool right_handle_hovered = ImGui::IsItemHovered();
		bool right_handle_active = ImGui::IsItemActive();

		if (right_handle_active && mouse_in_scrollbar) {
			auto old_start = frame_span.end;
			frame_span.end = std::clamp(old_start + mouse_delta_frames, frame_span.start, iseq->max_frame());
		}

		ImGui::SetCursorPos(left_handle_pos + handle_size);
		ImGui::InvisibleButton("middle_handle", ImVec2(right_handle_pos.x - left_handle_pos.x, handle_size.y));
		bool middle_handle_hovered = ImGui::IsItemHovered();
		bool middle_handle_active = ImGui::IsItemActive();

		if (middle_handle_active && mouse_in_scrollbar) {

			auto old_start = frame_span.start;
			auto old_end = frame_span.end;

			if (mouse_delta_frames > 0) {
				frame_span.end = std::clamp(old_end + mouse_delta_frames, frame_span.start, iseq->max_frame());
				auto real_delta = frame_span.end - old_end;
				frame_span.start = frame_span.start + real_delta;
			}
			else {
				frame_span.start = std::clamp(old_start + mouse_delta_frames, 0, frame_span.end);
				auto real_delta = frame_span.start - old_start;
				frame_span.end = frame_span.end + real_delta;
			}
		}

		draw.add_rect_filled({left_handle_rect.Max - ImVec2(handle_size.x / 2, 0), right_handle_rect.Min + ImVec2(handle_size.y / 2, 0)}, get_color((middle_handle_hovered || middle_handle_active) ? ImGuiCol_ScrollbarGrabActive : ImGuiCol_ScrollbarGrab));
		draw.add_rect_filled(left_handle_rect, get_color((left_handle_hovered || left_handle_active || middle_handle_active) ? ImGuiCol_ScrollbarGrabActive : ImGuiCol_ScrollbarGrab), handle_size.x - 4);
		draw.add_rect_filled(right_handle_rect, get_color((right_handle_hovered || right_handle_active || middle_handle_active) ? ImGuiCol_ScrollbarGrabActive : ImGuiCol_ScrollbarGrab), handle_size.x - 4);

		if (left_handle_hovered || right_handle_hovered || left_handle_active || right_handle_active) {
			ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
		}

		return frame_span;
	}

	std::string format_time(double seconds) {
		int minutes = (int)seconds / 60;
		int round_sec = std::floor((seconds - minutes * 60));
		int ms = std::floor((seconds - minutes * 60 - round_sec) * 1000);
		return std::format("{}:{:02}.{:03}", minutes, round_sec, ms);
	}

	bool show(interface* iseq, time_span::int_t& current_frame, int fps_snap) {
		auto& io = ImGui::GetIO();
		auto& style = ImGui::GetStyle();

		drawing_helper draw{
			.draw_list = ImGui::GetWindowDrawList(),
			.canvas_pos = ImGui::GetCursorScreenPos(),
			.canvas_size = ImGui::GetContentRegionAvail()
		};
		const auto mpos = io.MousePos - draw.canvas_pos;
		const bool window_focused = ImGui::IsWindowFocused();

		const auto canvas_size = ImGui::GetContentRegionAvail();

		const auto item_height = ImGui::GetTextLineHeight() * 1.2f;
		const auto legend_width = 200.0f;
		const auto timeline_width = canvas_size.x - legend_width;
		
		const auto frame_movement_size = time_span::framerate / fps_snap;

		draw.add_rect_filled(ImRect(ImVec2(0,0), draw.canvas_size), get_color(ImGuiCol_WindowBg), 0);
		draw.add_rect_filled(ImRect(ImVec2(0, 0), ImVec2(draw.canvas_size.x, item_height)), get_color(ImGuiCol_MenuBarBg), 0);

		auto old_cursor = ImGui::GetCursorPos();
		auto scrollbar_area = ImRect(ImVec2(legend_width, draw.canvas_size.y - item_height), canvas_size);
		auto frame_span = scrollbar_logic(io, iseq, draw, mpos, scrollbar_area, frame_movement_size);
		double pixels_per_frame = (timeline_width) / (frame_span.end - frame_span.start);

		auto mouse_delta_frames = io.MouseDelta.x / pixels_per_frame;

		auto span_info = std::format("{} - {}", format_time(frame_span.start / double(time_span::framerate)),
						format_time(frame_span.end / double(time_span::framerate))
				);

		draw.add_text({ 10, draw.canvas_size.y - item_height }, get_color(ImGuiCol_Text), span_info.c_str());

		ImGui::SetCursorPos(old_cursor);

		ImGui::SetCursorPos(ImGui::GetCursorPos() + ImVec2(0, item_height));
		auto child_pos = ImGui::GetCursorScreenPos();
		ImGui::BeginChild("##timeline_scroll", ImVec2(draw.canvas_size.x, draw.canvas_size.y - 2 * item_height));

		drawing_helper child_draw{
			.draw_list = ImGui::GetWindowDrawList(),
			.canvas_pos = child_pos,
		};

		// item names
		for (int tid = 0; tid < iseq->item_count(); tid++) {
			auto item_type = iseq->item_type(tid);
			auto item_name = iseq->item_name(tid);
			auto item_type_name = iseq->item_type_name(item_type);

			auto item_rect = ImRect(ImVec2(10, item_height * (tid)), ImVec2(canvas_size.x, item_height * (tid + 1)));
			auto item_text_pos = item_rect.Min;

			child_draw.add_text(item_text_pos, 0xFFFFFFFF, item_name.data());
		}

		ImGui::SetCursorPos({ legend_width, 0 });
		ImGui::BeginChild("##timeline", ImVec2({ canvas_size.x - legend_width, canvas_size.y - 2 * item_height }));

		drawing_helper timeline_draw{
			.draw_list = ImGui::GetWindowDrawList(),
			.canvas_pos = child_draw.canvas_pos + ImVec2(legend_width, 0),
			.canvas_size = ImVec2({ canvas_size.x - legend_width, canvas_size.y - 2 * item_height })
		};

		auto mpos_timeline = timeline_draw.to_local(ImGui::GetMousePos());
		auto mpos_frame = time_span::int_t{ -1 };

		if (mpos_timeline.x >= 0 && mpos_timeline.x <= timeline_draw.canvas_size.x && mpos_timeline.y >= 0 && mpos_timeline.y <= timeline_draw.canvas_size.y) {			
			mpos_frame = frame_span.start + (mpos_timeline.x / pixels_per_frame);
		}

		static bool dragging = false;
		static ImGuiID dragging_id = 0;
		static double drag_buildup = 0;

		auto current_active_id = ImGui::GetActiveID();

		if (!dragging && current_active_id) {
			dragging = true;
			drag_buildup = 0;
			dragging_id = current_active_id;
		}
		else if (dragging && !current_active_id) {
			dragging = false;
			drag_buildup = mouse_delta_frames;
			dragging_id = 0;
		}
		else if (dragging) {
			drag_buildup += mouse_delta_frames;
		}

		int payload_type = -1;
		if (ImGui::IsDragDropActive()) {
			auto payload = ImGui::GetDragDropPayload();
			for (int i = 0; i < iseq->item_type_count(); i++) {
				auto item_payload_name = iseq->drag_drop_payload_name(i);
				if (!item_payload_name.empty() && payload->IsDataType(item_payload_name.data())) payload_type = i;
			}
		}

		// draw sequences
		for (int tid = 0; tid < iseq->item_count(); tid++) {
			auto table_rect = ImRect(ImVec2(0, item_height * (tid)), ImVec2(canvas_size.x, item_height * (tid + 1)));
			timeline_draw.add_rect_filled(table_rect, get_color(tid % 2 == 0 ? ImGuiCol_TableRowBg : ImGuiCol_TableRowBgAlt));

			if (payload_type == iseq->item_type(tid)) {
				if (ImGui::BeginDragDropTargetCustom(timeline_draw.to_screen(table_rect), ImGui::GetID(std::format("{}", tid).c_str()))) {
					auto payload = ImGui::AcceptDragDropPayload(iseq->drag_drop_payload_name(payload_type).data());

					if (payload) {
						iseq->add_from_drag_drop(tid, mpos_frame, payload->Data);
					}
					else {
						auto local_pos = timeline_draw.to_local(ImGui::GetMousePos());
						timeline_draw.add_line({ local_pos.x, table_rect.Min.y }, { local_pos.x, table_rect.Max.y }, 0xFF0000FF, 2);
					}

					ImGui::EndDragDropTarget();
				}
			}

			for (int sid = 0; sid < iseq->item_segment_count(tid); sid++) {
				auto& segment = iseq->item_segment(tid, sid);
				
				if (segment.intersects(frame_span)) {
					auto seg_width = segment.duration() * pixels_per_frame;
					auto seg_start = (segment.start() - frame_span.start) * pixels_per_frame;

					auto seg_rect = ImRect(ImVec2(seg_start, item_height * (tid)), ImVec2(seg_start + seg_width, item_height * (tid + 1)));

					segment_logic(timeline_draw, segment, iseq->item_segment_name(tid, sid), std::format("##segment_{}_{}", tid, sid), seg_rect, frame_movement_size, dragging_id, drag_buildup);
				}
			}
		}

		ImGui::EndChild();
		ImGui::EndChild();

		draw.add_line({ legend_width, 0 }, { legend_width, canvas_size.y }, get_color(ImGuiCol_Border), style.WindowBorderSize);

		//draw_list->AddLine(ImVec2(canvas_pos.x + legend_width, canvas_pos.y + header_size.y), ImVec2(canvas_pos.x + legend_width, canvas_pos.y + canvas_size.y), get_color(ImGuiCol_Border), style.WindowBorderSize);

		return true;
	}
};