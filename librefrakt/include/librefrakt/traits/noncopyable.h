#pragma once

namespace rfkt::traits {

	class noncopyable {
	protected:
		noncopyable() = default;
		~noncopyable() = default;
		noncopyable(const noncopyable&) = delete;
		noncopyable& operator=(const noncopyable&) = delete;
	};

	static_assert(!std::copyable<noncopyable>);
}