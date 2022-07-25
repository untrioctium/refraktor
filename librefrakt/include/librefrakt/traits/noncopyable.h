#pragma once

namespace rfkt::traits {
	template<typename Class>
	class noncopyable {
	protected:
		noncopyable() = default;
		~noncopyable() = default;
		noncopyable(const noncopyable&) = delete;
		noncopyable& operator=(const noncopyable&) = delete;
	};
}