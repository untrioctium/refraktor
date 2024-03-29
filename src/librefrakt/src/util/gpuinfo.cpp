#include <optional>
#include <dylib.hpp>

#include "librefrakt/util/gpuinfo.h"

static inline std::optional<dylib> nvml_api = {};

using nvmlReturn_t = unsigned int;

enum nvmlTemperatureThresholds_t {
	NVML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0,
	NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1,
	NVML_TEMPERATURE_THRESHOLD_MEM_MAX = 2,
	NVML_TEMPERATURE_THRESHOLD_GPU_MAX = 3,
	NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN = 4,
	NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_CURR = 5,
	NVML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX = 6
};

enum nvmlClockType_t
{
	NVML_CLOCK_GRAPHICS = 0,
	NVML_CLOCK_SM = 1,
	NVML_CLOCK_MEM = 2,
	NVML_CLOCK_VIDEO = 3
};

enum nvmlClockId_t
{
	NVML_CLOCK_ID_CURRENT = 0,
	NVML_CLOCK_ID_APP_CLOCK_TARGET = 1,
	NVML_CLOCK_ID_APP_CLOCK_DEFAULT = 2,
	NVML_CLOCK_ID_CUSTOMER_BOOST_MAX = 3
};

static inline void(*nvmlInit_v2)() = nullptr;
static inline void(*nvmlShutdown)() = nullptr;

static inline nvmlReturn_t(*nvmlDeviceGetCount_v2)(unsigned int*) = nullptr;
static inline nvmlReturn_t(*nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*) = nullptr;
static inline nvmlReturn_t(*nvmlDeviceGetFanSpeed)(nvmlDevice_t, unsigned int*) = nullptr;

static inline nvmlReturn_t(*nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*) = nullptr;
static inline nvmlReturn_t(*nvmlDeviceGetEnforcedPowerLimit)(nvmlDevice_t, unsigned int*) = nullptr;

static inline nvmlReturn_t(*nvmlDeviceGetTemperatureThreshold)(nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int*) = nullptr;
static inline nvmlReturn_t(*nvmlDeviceGetTemperature)(nvmlDevice_t, unsigned int, unsigned int*) = nullptr;

static inline nvmlReturn_t(*nvmlDeviceGetClock)(nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int*) = nullptr;
static inline nvmlReturn_t(*nvmlDeviceGetMaxCustomerBoostClock)(nvmlDevice_t, nvmlClockType_t, unsigned int*) = nullptr;

static inline nvmlReturn_t(*nvmlDeviceGetTotalEnergyConsumption)(nvmlDevice_t, unsigned long long*) = nullptr;

static inline nvmlReturn_t(*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, rfkt::gpuinfo::memory_info*) = nullptr;

template<nvmlClockId_t Id, nvmlClockType_t Type>
unsigned int get_clock(nvmlDevice_t dev) noexcept {
	unsigned int value;
	nvmlDeviceGetClock(dev, Type, Id, &value);
	return value * 1'000'000;
}

void rfkt::gpuinfo::init() {
	struct init_wrapper {
		init_wrapper() {

			nvml_api = dylib{ "nvml" };

			if (!nvml_api) return;

			nvmlInit_v2 = nvml_api->get_function<void()>("nvmlInit_v2");
			nvmlShutdown = nvml_api->get_function<void()>("nvmlShutdown");

			nvmlDeviceGetCount_v2 = nvml_api->get_function<nvmlReturn_t(unsigned int*)>("nvmlDeviceGetCount_v2");
			nvmlDeviceGetHandleByIndex_v2 = nvml_api->get_function<nvmlReturn_t(unsigned int, nvmlDevice_t*)>("nvmlDeviceGetHandleByIndex_v2");
			nvmlDeviceGetFanSpeed = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, unsigned int*)>("nvmlDeviceGetFanSpeed");

			nvmlDeviceGetPowerUsage = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, unsigned int*)>("nvmlDeviceGetPowerUsage");
			nvmlDeviceGetEnforcedPowerLimit = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, unsigned int*)>("nvmlDeviceGetEnforcedPowerLimit");

			nvmlDeviceGetTemperatureThreshold = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int*)>("nvmlDeviceGetTemperatureThreshold");
			nvmlDeviceGetTemperature = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, unsigned int, unsigned int*)>("nvmlDeviceGetTemperature");

			nvmlDeviceGetClock = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int*)>("nvmlDeviceGetClock");
			nvmlDeviceGetMaxCustomerBoostClock = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, nvmlClockType_t, unsigned int*)>("nvmlDeviceGetMaxCustomerBoostClock");

			nvmlDeviceGetTotalEnergyConsumption = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, unsigned long long*)>("nvmlDeviceGetTotalEnergyConsumption");

			nvmlDeviceGetMemoryInfo = nvml_api->get_function<nvmlReturn_t(nvmlDevice_t, rfkt::gpuinfo::memory_info*)>("nvmlDeviceGetMemoryInfo");

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
	nvmlDeviceGetHandleByIndex_v2(index, &dev);
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
	nvmlDeviceGetTemperature(dev, 0, &degrees);
	return degrees;
}

auto rfkt::gpuinfo::device::temperature_percent() const -> unsigned int {
	return temperature() * 100 / max_temperature();
}

auto rfkt::gpuinfo::device::clock() const -> unsigned int {
	return static_cast<std::size_t>(get_clock<NVML_CLOCK_ID_CURRENT, NVML_CLOCK_GRAPHICS>(dev));
}

auto rfkt::gpuinfo::device::max_clock() const -> unsigned int {
	unsigned int value;
	const auto ret = nvmlDeviceGetMaxCustomerBoostClock(dev, NVML_CLOCK_SM, &value);
	return value;
}

auto rfkt::gpuinfo::device::clock_percent() const -> unsigned int {
	return clock() * 100 / max_clock();
}

auto rfkt::gpuinfo::device::memory_info() const -> gpuinfo::memory_info
{
	auto info = gpuinfo::memory_info{};
	auto ret = nvmlDeviceGetMemoryInfo(dev, &info);
	return info;
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