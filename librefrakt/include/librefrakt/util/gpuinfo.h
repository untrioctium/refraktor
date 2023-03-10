#include <nvml.h>

namespace rfkt::gpuinfo {

	inline void init() {
		struct init_wrapper {
			init_wrapper() {
				nvmlInit_v2();
			}

			~init_wrapper() {
				nvmlShutdown();
			}
		};

		static init_wrapper s{};
	}

	class device {
	public:

		static auto count() -> unsigned int {
			unsigned int count;
			nvmlDeviceGetCount_v2(&count);
			return count;
		}

		static auto by_index(unsigned int index) -> device {
			if (index >= count()) return {};
			nvmlDevice_t dev;
			nvmlDeviceGetHandleByIndex(index, &dev);
			return device{ dev };
		}

		auto fan_speed()  const -> unsigned int {
			unsigned int speed;
			nvmlDeviceGetFanSpeed(dev, &speed);
			return speed;
		}

		auto wattage()  const -> unsigned int {
			unsigned int milliwatts;
			nvmlDeviceGetPowerUsage(dev, &milliwatts);
			return milliwatts / 1000;
		}

		auto max_wattage()  const -> unsigned int {
			unsigned int milliwatts;
			nvmlDeviceGetEnforcedPowerLimit(dev, &milliwatts);
			return milliwatts / 1000;
		}

		auto wattage_percent() const  -> unsigned int {
			return wattage() * 100 / max_wattage();
		}

		auto max_temperature()  const -> unsigned int {
			unsigned int degrees;
			nvmlDeviceGetTemperatureThreshold(dev, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &degrees);
			return degrees;
		}

		auto temperature()  const {
			unsigned int degrees;
			nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &degrees);
			return degrees;
		}

		auto temperature_percent() const {
			return temperature() * 100 / max_temperature();
		}

		auto clock() const {
			return static_cast<std::size_t>(get_clock<NVML_CLOCK_ID_CURRENT, NVML_CLOCK_GRAPHICS>());
		}

		auto max_clock() const {
			unsigned int value;
			const auto ret = nvmlDeviceGetMaxCustomerBoostClock(dev, NVML_CLOCK_SM, &value);
			return value;
		}

		auto clock_percent() const {
			return clock() * 100 / max_clock();
		}

		// returns the total amount of energy used by the GPU in joules since the driver was loaded
		// use two calls to this function to get the energy used in joules over a period of time
		auto total_energy_usage() const {
			unsigned long long millijoules;
			nvmlDeviceGetTotalEnergyConsumption(dev, &millijoules);
			return static_cast<double>(millijoules) / 1000.0;
		}

		bool valid() const {
			return dev != nullptr;
		}

		device() = default;


	private:
		explicit device(nvmlDevice_t dev) : dev(dev) {}
		mutable nvmlDevice_t dev = nullptr;

		template<nvmlClockId_t Id, nvmlClockType_t Type>
		unsigned int get_clock() const noexcept {
			unsigned int value;
			const auto ret = nvmlDeviceGetClock(dev, Type, Id, &value);
			return value * 1'000'000;
		}
	};

	class power_meter {
	public:
		explicit(false) power_meter(device dev) : dev(dev) { reset(); };

		void reset() {
			last_time = clock::now();
			last_energy_usage = dev.total_energy_usage();
		}

		auto elapsed_seconds() const {
			return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - last_time).count() / 1000.0;
		}

		auto joules() const {
			return dev.total_energy_usage() - last_energy_usage;
		}

		auto watts() const {
			return joules() / elapsed_seconds();
		}

	private:
		using clock = std::chrono::steady_clock;
		using measurement_type = decltype(device{}.total_energy_usage());

		device dev;

		decltype(clock::now()) last_time = clock::now();
		measurement_type last_energy_usage = 0;
	};

}