#include <optional>
#include <vector>

class time_span {
public:
	using int_t = int;
	struct range {
		int_t start;
		int_t end;

		int_t length() const noexcept { return end - start; }
		bool contains(int_t val) const noexcept { return val >= start && val <= end; }
	};

	static constexpr int framerate = 3600;

	constexpr time_span(time_span&& o) noexcept = default;
	constexpr time_span(const time_span& o) noexcept = default;
	constexpr time_span& operator=(time_span&& o) noexcept = default;
	constexpr time_span& operator=(const time_span& o) noexcept = default;

	constexpr int_t duration() const noexcept { return end_frame - start_frame; }

	constexpr int_t start() const noexcept { return start_frame; }
	constexpr int_t end() const noexcept { return end_frame; }

	constexpr static double seconds_to_frames(double seconds) noexcept {
		return seconds * framerate;
	}

	constexpr double duration_seconds() const noexcept {
		return duration() / double(framerate);
	}

	constexpr double start_seconds() const noexcept {
		return start_frame / double(framerate);
	}

	constexpr double end_seconds() const noexcept {
		return end_frame / double(framerate);
	}

	constexpr int_t intersects(const time_span& o) const noexcept {
		return intersects(range{ o.start(), o.end() });
	}

	constexpr int_t intersects(int_t start, int_t end) const noexcept {
		return intersects(range{ start, end });
	}

	constexpr int_t intersects(const range& o) const noexcept {
		return contains(o.start) || contains(o.end) || o.contains(start_frame) || o.contains(end_frame);
	}

	constexpr int_t contains(int_t frame) const noexcept {
		return frame >= start_frame && frame <= end_frame;
	}

	constexpr int_t adjust_start(int_t delta) noexcept {
		auto new_val = start_frame + delta;
		if (new_val > end_frame) {
			new_val = end_frame;
		}
		else if (new_val < left_barrier) {
			new_val = left_barrier;
		}

		auto actual_delta = new_val - start_frame;
		start_frame = new_val;
		return actual_delta;
	}

	constexpr int_t adjust_end(int_t delta) noexcept {
		auto new_val = end_frame + delta;
		if (new_val < start_frame) {
			new_val = start_frame;
		}
		else if (new_val > right_barrier) {
			new_val = right_barrier;
		}

		auto actual_delta = new_val - end_frame;
		end_frame = new_val;
		return actual_delta;
	}

	constexpr int_t move(int_t delta) noexcept {
		if (delta < 0) {
			return adjust_end(adjust_start(delta));
		}
		else {
			return adjust_start(adjust_end(delta));
		}
	}

	constexpr std::optional<std::pair<int_t, int_t>> intersection(const time_span& o) const noexcept {
		if (!intersects(o)) {
			return std::nullopt;
		}

		return std::pair<int_t, int_t>{
			std::max(start_frame, o.start_frame),
			std::min(end_frame, o.end_frame)
		};
	}

	constexpr int_t min_value() const noexcept {
		return left_barrier;
	}

	constexpr int_t max_value() const noexcept {
		return right_barrier;
	}

private:

	time_span() = default;
	constexpr time_span(int_t a, int_t b) noexcept : start_frame(std::min(a, b)), end_frame(std::max(a, b)) {}

	template<typename T>
	friend class track;

	mutable int_t left_barrier = 0;
	mutable int_t right_barrier = std::numeric_limits<int_t>::max();
	int_t start_frame = 0;
	int_t end_frame = 0;
};

using guid_t = std::pair<std::uint64_t, std::uint64_t>;

struct flame_track_data {
	guid_t id;
};

template<typename Data>
class track {
public:

	struct value_t {
		time_span span;
		Data data;
	};

	value_t* find_segment(this auto&& self, time_span::int_t time) noexcept {
		for (int i = 0; i < self.segments.size(); i++) {
			auto& seg = self.segments[i];
			if (seg.span.start() > time) return nullptr;
			if (seg.span.contains(time)) {
				adjust_barriers(i);
				return &seg;
			}

			return nullptr;
		}
	}

	/*auto find_adjacent(this auto&& self, time_span::int_t time) noexcept {
		for (int i = 0; i < self.segments.size(); i++) {
			auto& seg = self.segments[i];
			if (seg.span.contains(time)) {
				self.adjust_barriers(i);
				return std::span{ self.segments.begin() + i, self.segments.begin() + i + 1 };
			}
			if(seg.span.start_frame)
		}
	}*/

	auto segment_count() const noexcept { return segments.size(); }
	auto begin(this auto&& self) { return self.segments.begin(); }
	auto end(this auto&& self) { return self.segments.end(); }
	auto& operator[](this auto&& self, std::size_t idx) {
		self.adjust_barriers(idx);
		return self.segments[idx];
	}

	void insert_segment(time_span::int_t start, time_span::int_t end, Data&& data) {
		// TODO: Better insert
		segments.emplace_back(value_t{ time_span{ start, end }, std::move(data) });
		sort_segments();
	}

	bool space_is_free(int a, int b) const noexcept {
		return find_segment(std::min(a, b)) == nullptr && find_segment(std::max(a, b)) == nullptr;
	}

private:

	void adjust_barriers(int i) const {
		auto& seg = segments[i];
		seg.span.left_barrier = (i == 0) ? 0 : segments[i - 1].span.end_frame;
		seg.span.right_barrier = (i == segments.size() - 1) ? std::numeric_limits<time_span::int_t>::max() : segments[i + 1].span.start_frame;
	}

	void sort_segments() {
		std::sort(segments.begin(), segments.end(), [](const value_t& l, const value_t& r) {
			return l.span.start() < l.span.start();
		});
	}

	std::vector<value_t> segments;
};

namespace rfkt::gui::panel::timeline {

	struct interface {

		virtual time_span::int_t min_frame() const = 0;
		virtual time_span::int_t max_frame() const = 0;

		virtual int item_count() const = 0;
		virtual int item_type_count() const = 0;
		virtual std::string_view item_type_name(int id) const = 0;
		virtual std::string_view item_name(int id) const = 0;

		virtual int item_type(int id) const = 0;
		virtual int item_segment_count(int id) const = 0;
		virtual time_span& item_segment(int item_id, int seg_id) = 0;

		virtual ~interface() = default;
	};

	bool show(interface* iseq, time_span::int_t& current_frame, int fps_snap = time_span::framerate);
}

using flame_track = track<flame_track_data>;