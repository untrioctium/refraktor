#pragma once

#include <chrono>

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
}