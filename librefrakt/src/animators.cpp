#include <librefrakt/animators.h>
#define _USE_MATH_DEFINES
#include <math.h>

namespace rfkt::animators {

	class sine: public animator::registrar<sine> {
	public:
		sine(json data) :
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
		interpolate(json data):
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
		increase(json data) :
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
}

void rfkt::animator::init_builtins()
{
	animators::sine::registerT();
	animators::interpolate::registerT();
	animators::increase::registerT();
}