#include <librefrakt/animators.h>
#define _USE_MATH_DEFINES
#include <math.h>

namespace rfkt::animators {

	class sine: public animator::registrar<sine> {
	public:
		sine(const json& data) :
			frequency(data.value("frequency", 1.0)),
			amplitude(data.value("amplitude", 1.0)),
			sharpness(data.value("sharpness", 0)),
			absolute(data.value("use_absolute", false))
		{}

		auto serialize() const -> json {
			auto o = json::object();
			o.emplace("frequency", frequency);
			o.emplace("amplitude", amplitude);
			o.emplace("absolute", absolute);
			o.emplace("sharpness", sharpness);
			return o;
		}

		auto apply(double t, double initial) const -> double {
			auto v = sin(t * frequency * M_PI * 2.0);
			if (sharpness > 0) v = copysign(1.0, v) * pow(abs(v), sharpness);
			return initial + ((absolute)? abs(v): v) * amplitude;
		}

		auto name() const -> std::string { return "sine"; }

	private:
		double frequency;
		double amplitude;
		int sharpness;
		bool absolute;
	};

	class interpolate : public animator::registrar<interpolate> {
	public:
		interpolate(const json& data):
			smooth(data.value("smooth", false)),
			final_value(data.value("final_value", 0.0))
		{}

		auto serialize() const -> nlohmann::json {
			auto o = json::object();
			o.emplace("final_value", final_value);
			o.emplace("smooth", smooth);
			return o;
		}

		auto apply(double t, double initial) const -> double {
			if (t <= 0.0) return initial;
			if (t >= 1.0) return final_value;

			if (smooth) t = -2.0 * t * t * t + 3.0 * t * t;
			return initial * (1.0 - t) + t * final_value;
		}

		auto name() const -> std::string { return "interpolate"; }
	private:
		bool smooth;
		double final_value;
	};

	class increase : public animator::registrar<increase> {
	public:
		increase(const json& data) :
			per_loop(data.value("per_loop", 360.0)) {}

		auto serialize() const -> json {
			auto o = json::object();
			o.emplace("per_loop", per_loop);
			return o;
		}

		auto apply(double t, double initial) const -> double {
			return initial + t * per_loop;
		}

		auto name() const -> std::string { return "increase"; }

	private:
		double per_loop;
	};

	class interp_children : public animator::registrar<interp_children> {
	public:
		interp_children(const json& data) :
			left(animator::make(data["left_name"], data["left"])),
			right(animator::make(data["right_name"], data["right"])),
			right_iv(data["right_iv"])
		{}

		auto serialize() const -> json {
			return json::object({
				{"left_name", left->name()},
				{"right_name", right->name()},
				{"left", left->serialize()},
				{"right", right->serialize(),
				{"right_iv"}, right_iv}
			});
		}

		auto apply(double t, double initial) const -> double {
			auto left_v = left->apply(t, initial);
			auto right_v = right->apply(t, right_iv);

			t = -2.0 * t * t * t + 3.0 * t * t;
			return left_v * (1.0 - t) + t * right_v;
		}

		auto name() const -> std::string { return "interp_children"; }

	private:
		std::unique_ptr<animator> left;
		std::unique_ptr<animator> right;
		double right_iv;
	};

	class noop : public animator::registrar<noop> {
	public:
		noop(const json& data) {}

		auto serialize() const -> json { return json::object(); }
		auto apply(double t, double initial) const -> double { return initial; }
		auto name() const -> std::string { return "noop"; }

	};
}

void rfkt::animator::init_builtins()
{
	animators::sine::registerT();
	animators::interpolate::registerT();
	animators::increase::registerT();
}