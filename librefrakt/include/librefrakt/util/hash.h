#pragma once
#include <string_view>
#include <span>

#include <librefrakt/traits/noncopyable.h>

namespace rfkt {
	class hash_t {
	public:

		hash_t() = default;
		hash_t(std::size_t low, std::size_t high) : bytes(low, high) {}

		bool operator == (const rfkt::hash_t& o) const noexcept = default;
		auto operator <=> (const rfkt::hash_t& o) const noexcept = default;

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

		template<typename Contained>
		void update(std::span<Contained> sp) {
			update(sp.data(), sp.size_bytes());
		}

		void update(std::string_view s) {
			update(s.data(), s.size());
		}

		void update(std::integral auto value) {
			update(&value, sizeof(value));
		}

	private:
		void* state_;
	};

	auto calc( const void*, std::size_t ) -> hash_t;

	inline auto calc( std::string_view str ) {
		return calc(str.data(), str.size());
	}

	template<typename Contained>
	auto calc( const std::span<Contained> sp ) {
		return calc(sp.data(), sp.size_bytes());
	}
}