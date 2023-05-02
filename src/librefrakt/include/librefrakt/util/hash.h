#pragma once
#include <string_view>
#include <span>

#include <librefrakt/traits/noncopyable.h>

namespace rfkt {
	class hash_t {
	public:
		using value_type = std::size_t;

		constexpr hash_t() = default;
		constexpr hash_t(value_type low, value_type high) : bytes(low, high) {}

		constexpr auto operator <=> (const rfkt::hash_t& o) const noexcept = default;

		auto str16() const noexcept ->std::string;
		auto str32() const noexcept ->std::string;
		auto str64() const noexcept ->std::string;

	private:
		std::pair<value_type, value_type> bytes = { 0, 0 };
	};

	static_assert(std::move_constructible<hash_t>);
}

namespace rfkt::hash {

	namespace detail {
		template<typename T>
		struct is_std_array : std::false_type {};

		template<typename T, std::size_t N>
		struct is_std_array<std::array<T, N>> : std::true_type {};
	}

	template<typename T>
	concept numeric_type = std::integral<T> || std::floating_point<T>;

	template<typename T>
	concept numeric_std_array = detail::is_std_array<T>::value && numeric_type<typename T::value_type>;

	template<typename T>
	concept hashable_type = numeric_type<T> || numeric_std_array<T>;

	class state_t: public traits::noncopyable {
	public:	
		state_t() noexcept;
		~state_t() noexcept;
	
		state_t(state_t&& o) noexcept {
			std::swap(state_, o.state_);
		}

		state_t& operator=(state_t&& o) noexcept {
			std::swap(state_, o.state_);
			return *this;
		}

		auto digest() const noexcept -> hash_t;

		template<hashable_type Contained>
		void update(std::span<Contained> sp) noexcept {
			update(sp.data(), sp.size_bytes());
		}

		template<hashable_type Contained>
		void update(const std::vector<Contained>& v) noexcept {
			update(v.data(), v.size() * sizeof(Contained));
		}

		void update(std::string_view s) noexcept {
			update(s.data(), s.size());
		}

		template<hashable_type T>
		void update(T value) noexcept {
			update(&value, sizeof(value));
		}

	private:

		void update(const void*, std::size_t) noexcept;

		void* state_ = nullptr;
	};

	template<typename... Args>
	rfkt::hash_t calc(Args&&... args) noexcept {
		auto state = rfkt::hash::state_t{};
		state.update(std::forward<Args>(args)...);
		return state.digest();
	}
}