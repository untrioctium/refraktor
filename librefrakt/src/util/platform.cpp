#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <processthreadsapi.h>
#endif

#include <librefrakt/util/platform.h>

auto rfkt::platform::dynlib::load(const std::string& libname) -> std::unique_ptr<dynlib>
{
	auto lib = std::unique_ptr<dynlib>(new dynlib());

#ifdef _WIN32
	lib->handle = LoadLibraryA(libname.c_str());
#else
	static_assert(false, "No platform handler");
#endif

	return lib;
}

rfkt::platform::dynlib::~dynlib()
{
	if (handle == nullptr) return;

#ifdef _WIN32
	FreeLibrary((HMODULE) handle);
#endif

}

void* rfkt::platform::dynlib::symbol_impl(const std::string& name)
{
	if (handle == nullptr) return nullptr;

#ifdef _WIN32
	return GetProcAddress((HMODULE) handle, name.c_str());
#else
	static_assert(false, "No platform handler");
#endif
}

std::string rfkt::platform::show_open_dialog()
{
#ifdef _WIN32
	OPENFILENAMEA open;
	memset(&open, 0, sizeof(open));

	char filename[256];
	memset(&filename, 0, sizeof(filename));
	open.lStructSize = sizeof(open);
	open.lpstrFile = filename;
	std::string path = (std::filesystem::current_path() / "assets" / "flames").string();
	open.lpstrInitialDir = path.c_str();
	//open.hwndOwner = glfwGetWin32Window(glfwGetCurrentContext());
	open.nMaxFile = sizeof(filename);
	open.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

	if (GetOpenFileNameA(&open) == TRUE) return open.lpstrFile;
	else
	{	
		//SPDLOG_ERROR("Dialog error: 0x{:04x}", CommDlgExtendedError());
		return {};
	}
#endif
}

void rfkt::platform::set_thread_name(const std::wstring& name) {
	SetThreadDescription(GetCurrentThread(), name.c_str());
}

namespace rfkt::platform {

	class process::impl_t {
	public:
		impl_t(const std::string& cmd, const std::string& args) {
#ifdef _WIN32
			std::memset(&si, 0, sizeof(si));
			si.cb = sizeof(si);
			std::memset(&pi, 0, sizeof(pi));

			valid = CreateProcess(
				(cmd.length() > 0) ? const_cast<char*>(cmd.c_str()) : nullptr,
				(args.length() > 0) ? const_cast<char*>(args.c_str()) : nullptr,
				nullptr, nullptr, false, 0, nullptr, nullptr, &si, &pi);
#endif		
		}

		bool running() {
#ifdef _WIN32
			return valid;
#endif
		}
		auto wait_for_exit() -> int {
#ifdef _WIN32
			if (!valid) return -1;

			WaitForSingleObject(pi.hProcess, INFINITE);
			exit_code = (GetExitCodeProcess(pi.hProcess, &exit_code) != 0)? 0: 1;
			CloseHandle(pi.hProcess);
			CloseHandle(pi.hThread);
			valid = false;
			return static_cast<int>(exit_code);
#endif
		}

	private:
#ifdef _WIN32
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
		bool valid = false;
		DWORD exit_code = 0;
#endif
	};

	process::process() = default;
	process::~process() = default;

	process::process(const std::string& cmd, const std::string& args) : impl(new impl_t{ cmd, args}) {}

	bool process::running() { return impl && impl->running(); }
	auto process::wait_for_exit() -> int {
		if (!impl) return -1;

		return impl->wait_for_exit();
	}
}