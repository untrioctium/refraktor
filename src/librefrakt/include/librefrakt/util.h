#pragma once

#include <chrono>
#include <random>
#include <format>
#include <span>

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...)->overloaded<Ts...>;

namespace rfkt {
	class timer {
	public:

		timer() { reset(); }

		auto count() {
			return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
		}

		void reset() {
			start = std::chrono::high_resolution_clock::now();
		}

	private:
		decltype(std::chrono::high_resolution_clock::now()) start;
	};

	class guid {
	public:
		guid() = default;
		guid(const guid&) = default;
		guid(guid&&) = default;
		guid& operator=(const guid&) = default;
		guid& operator=(guid&&) = default;

		std::strong_ordering operator<=>(const guid& other) const {
			return data <=> other.data;
		}

		bool operator==(const guid& other) const {
			return data == other.data;
		}

		std::string to_string() const {
			return std::format(
				"{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
				data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], 
				data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]
			);
		}

		std::span<const unsigned char, 16> bytes() const {
			return data;
		}

	private:
		using data_t = std::array<unsigned char, 16>;
		
		data_t data = random();

		static auto random() -> data_t {
			thread_local std::random_device rd;
			thread_local std::mt19937 gen(rd());
			thread_local std::uniform_int_distribution dis(0, 255);

			data_t ret{};
			for (auto& i : ret) i = dis(gen);
			return ret;
		}
	};
}