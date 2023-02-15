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
			return { dev };
		}

		auto wattage() -> unsigned int {
			unsigned int milliwatts;
			nvmlDeviceGetPowerUsage(dev, &milliwatts);
			return milliwatts / 1000;
		}

		auto temperature() {
			unsigned int degrees;
			nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &degrees);
			return degrees;
		}

		auto clock() {
			return static_cast<std::size_t>(get_clock<NVML_CLOCK_ID_CURRENT, NVML_CLOCK_GRAPHICS>());
		}

		auto max_clock() {
			unsigned int value;
			const auto ret = nvmlDeviceGetMaxCustomerBoostClock(dev, NVML_CLOCK_SM, &value);
			return value;
		}

		bool valid() const {
			return dev != nullptr;
		}

		device() = default;


	private:
		device(nvmlDevice_t dev) : dev(dev) {}
		nvmlDevice_t dev = nullptr;

		template<nvmlClockId_t Id, nvmlClockType_t Type>
		unsigned int get_clock() noexcept {
			unsigned int value;
			const auto ret = nvmlDeviceGetClock(dev, Type, Id, &value);
			return value * 1'000'000;
		}
	};

}