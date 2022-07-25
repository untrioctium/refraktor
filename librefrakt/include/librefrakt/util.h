#pragma once

#include <chrono>
namespace chrono = std::chrono;

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...)->overloaded<Ts...>;

template<typename Func>
auto time_it(Func&& func) {
	using return_type = decltype(func());

	if constexpr (std::is_same_v<void, return_type>) {
		auto start = std::chrono::high_resolution_clock::now();
		func();
		auto end = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1'000'000.0;
	}
	else {
		auto start = std::chrono::high_resolution_clock::now();
		auto ret = func();
		auto end = std::chrono::high_resolution_clock::now();
		return std::pair<double, return_type>{ std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1'000'000.0, std::move(ret)};
	}
}

template<typename Contained>
class scroll_buffer {
public:
	scroll_buffer(std::size_t size) {
		max_size_ = size;
	}

	auto max_size() const { return max_size_; }
	auto size() const { return buffer.size(); }
	auto begin() const { return buffer.begin(); }
	auto end() const { return buffer.end(); }
	auto begin() { return buffer.begin(); }
	auto end() { return buffer.end(); }
	auto data() { return buffer.data(); }
	auto push(Contained&& element) {
		if (buffer.size() == max_size_) {
			buffer.erase(buffer.begin());
		}
		return buffer.emplace_back(std::move(element));
	}
private:
	std::vector<Contained> buffer;
	std::size_t max_size_;
};

namespace rfkt {

	template<typename T>
	struct vec2 {
		T x;
		T y;
	};

	using vec2f = vec2<float>;
	using vec2d = vec2<double>;
	using vec2i = vec2<int>;
	using vec2ui = vec2<unsigned int>;

	using byte_vector = std::vector<char>;

	/*inline auto to_base16(const byte_vector& bytes) -> std::string {
		auto ret = std::string{};
		ret.reserve(bytes.size() * 2);
		for (auto byte : bytes) {
			ret += fmt::format("{:02X}", static_cast<unsigned char>(byte));
		}
		return ret;
	}*/
}