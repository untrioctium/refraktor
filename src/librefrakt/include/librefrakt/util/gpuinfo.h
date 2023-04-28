#include <nvml.h>
#include <chrono>

namespace rfkt::gpuinfo {

	void init();

	class device {
	public:

		static auto count() -> unsigned int;
		static auto by_index(unsigned int index) -> device;

		auto fan_speed()  const -> unsigned int;

		auto wattage()  const -> unsigned int;
		auto max_wattage()  const -> unsigned int;
		auto wattage_percent() const -> unsigned int;

		auto max_temperature() const -> unsigned int;
		auto temperature() const -> unsigned int;
		auto temperature_percent() const -> unsigned int;

		auto clock() const -> unsigned int;
		auto max_clock() const -> unsigned int;
		auto clock_percent() const -> unsigned int;

		// returns the total amount of energy used by the GPU in joules since the driver was loaded
		// use two calls to this function to get the energy used in joules over a period of time
		auto total_energy_usage() const -> double;

		bool valid() const;

		device() = default;


	private:
		explicit device(nvmlDevice_t dev) : dev(dev) {}
		mutable nvmlDevice_t dev = nullptr;

		template<nvmlClockId_t Id, nvmlClockType_t Type>
		unsigned int get_clock() const noexcept {
			unsigned int value;
			nvmlDeviceGetClock(dev, Type, Id, &value);
			return value * 1'000'000;
		}
	};

	class power_meter {
	public:
		explicit(false) power_meter(device dev) : dev(dev) { reset(); };

		void reset();

		auto elapsed_seconds() const -> double;
		auto joules() const -> double;
		auto watts() const -> double;

	private:
		using clock = std::chrono::steady_clock;
		using measurement_type = decltype(device{}.total_energy_usage());

		device dev;

		decltype(clock::now()) last_time = clock::now();
		measurement_type last_energy_usage = 0;
	};

}