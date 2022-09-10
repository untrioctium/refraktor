#pragma once
#include <vector>
#include <string>
#include <array>

#include <librefrakt/traits/noncopyable.h>

namespace rfkt {
	class hash_t {
	public:

		hash_t() = default;
		hash_t(std::size_t low, std::size_t high) : bytes(low, high) {}

		bool operator == (const rfkt::hash_t& o) const noexcept {
			return bytes.first == o.bytes.first && bytes.second == o.bytes.second;
		}

		auto operator <=> (const rfkt::hash_t& o) const noexcept {
			return bytes <=> o.bytes;
		}

		auto str16() const->std::string;
		auto str32() const->std::string;
		auto str64() const->std::string;

	private:
		std::pair<std::size_t, std::size_t> bytes;
	};

	static_assert(std::move_constructible<hash_t>);
}

namespace rfkt::hash {

	class state_t: traits::noncopyable {
	public:	
		state_t();
		~state_t();
	
		auto digest() const -> hash_t;
		void update(const void*, std::size_t);
		void update(const std::string& s) {
			update(s.data(), s.size());
		}

		template<typename Contained>
		void update(const std::vector<Contained>& vec) {
			update(vec.data(), vec.size() * sizeof(Contained));
		}

		void update(std::integral auto value) {
			update(&value, sizeof(value));
		}

	private:
		void* state_;
	};

	auto calc( const void*, std::size_t ) -> hash_t;

	inline auto calc( const std::string& str ) {
		return calc(str.c_str(), str.size());
	}

	template<typename Contained, std::size_t Size>
	auto calc( const std::array<Contained, Size>& arr ) {
		return calc(arr.data(), Size * sizeof(Contained));
	}

	template<typename Contained>
	auto calc( const std::vector<Contained>& vec ) {
		return calc(vec.data(), vec.size() * sizeof(Contained));
	}
}