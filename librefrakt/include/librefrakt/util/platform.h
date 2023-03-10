#pragma once

#include <string>
#include <memory>

namespace rfkt::platform {

	class dynlib {
	public:
		static std::unique_ptr<dynlib> load(const std::string& libname);

		template<typename Func>
		Func symbol(const std::string& name) {
			return (Func)symbol_impl(name);
		}

		~dynlib();

	private:
		void* handle;
		void* symbol_impl(const std::string& name);
	};

	constexpr bool is_posix() {
#ifdef _WIN32
		return false;
#else
		return true;
#endif
	}

	std::string show_open_dialog();
	void set_thread_name(const std::wstring& name);

	class process {
	public:
		process(const std::string& cmd, const std::string& args);
		process();
		~process();

		process(const process&) = delete;
		process& operator=(const process&) = delete;

		process(process&&) = default;
		process& operator=(process&&) = default;

		bool running();
		auto wait_for_exit() -> int;

	private:
		struct impl_t;
		std::unique_ptr<impl_t> impl;
	};
}