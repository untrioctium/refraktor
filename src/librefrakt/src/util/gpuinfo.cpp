#include "librefrakt/util/gpuinfo.h"

void rfkt::gpuinfo::init() {
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

auto rfkt::gpuinfo::device::count() -> unsigned int {
	unsigned int count;
	nvmlDeviceGetCount_v2(&count);
	return count;
}

auto rfkt::gpuinfo::device::by_index(unsigned int index) -> device {
	if (index >= count()) return {};
	nvmlDevice_t dev;
	nvmlDeviceGetHandleByIndex(index, &dev);
	return device{ dev };
}

auto rfkt::gpuinfo::device::fan_speed() const -> unsigned int {
	unsigned int speed;
	nvmlDeviceGetFanSpeed(dev, &speed);
	return speed;
}

auto rfkt::gpuinfo::device::wattage() const -> unsigned int {
	unsigned int milliwatts;
	nvmlDeviceGetPowerUsage(dev, &milliwatts);
	return milliwatts / 1000;
}

auto rfkt::gpuinfo::device::max_wattage() const -> unsigned int {
	unsigned int milliwatts;
	nvmlDeviceGetEnforcedPowerLimit(dev, &milliwatts);
	return milliwatts / 1000;
}

auto rfkt::gpuinfo::device::wattage_percent() const -> unsigned int {
	return wattage() * 100 / max_wattage();
}

auto rfkt::gpuinfo::device::max_temperature() const -> unsigned int {
	unsigned int degrees;
	nvmlDeviceGetTemperatureThreshold(dev, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &degrees);
	return degrees;
}

auto rfkt::gpuinfo::device::temperature() const -> unsigned int {
	unsigned int degrees;
	nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &degrees);
	return degrees;
}

auto rfkt::gpuinfo::device::temperature_percent() const -> unsigned int {
	return temperature() * 100 / max_temperature();
}

auto rfkt::gpuinfo::device::clock() const -> unsigned int {
	return static_cast<std::size_t>(get_clock<NVML_CLOCK_ID_CURRENT, NVML_CLOCK_GRAPHICS>());
}

auto rfkt::gpuinfo::device::max_clock() const -> unsigned int {
	unsigned int value;
	const auto ret = nvmlDeviceGetMaxCustomerBoostClock(dev, NVML_CLOCK_SM, &value);
	return value;
}

auto rfkt::gpuinfo::device::clock_percent() const -> unsigned int {
	return clock() * 100 / max_clock();
}

// returns the total amount of energy used by the GPU in joules since the driver was loaded
// use two calls to this function to get the energy used in joules over a period of time

auto rfkt::gpuinfo::device::total_energy_usage() const -> double {
	unsigned long long millijoules;
	nvmlDeviceGetTotalEnergyConsumption(dev, &millijoules);
	return static_cast<double>(millijoules) / 1000.0;
}

bool rfkt::gpuinfo::device::valid() const {
	return dev != nullptr;
}

void rfkt::gpuinfo::power_meter::reset() {
	last_time = clock::now();
	last_energy_usage = dev.total_energy_usage();
}

auto rfkt::gpuinfo::power_meter::elapsed_seconds() const -> double {
	return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - last_time).count() / 1000.0;
}

auto rfkt::gpuinfo::power_meter::joules() const -> double {
	return dev.total_energy_usage() - last_energy_usage;
}

auto rfkt::gpuinfo::power_meter::watts() const -> double {
	return joules() / elapsed_seconds();
}
