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

	constexpr inline bool is_posix() {
#ifdef _WIN32
		return false;
#else
		return true;
#endif
	}

	std::string show_open_dialog();
	void set_thread_name(const std::wstring& name);
}