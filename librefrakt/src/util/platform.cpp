#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#include <processthreadsapi.h>
#else
#include <dlfcn.h>
#endif

#include <librefrakt/util/platform.h>

auto rfkt::platform::dynlib::load(const std::string& libname) -> std::unique_ptr<dynlib>
{
	auto lib = std::unique_ptr<dynlib>(new dynlib());

#ifdef _WIN32
	lib->handle = LoadLibraryA(libname.c_str());
#else
	lib->handle = dlopen(libname.c_str(), RTLD_NOW);
#endif

	return lib;
}

rfkt::platform::dynlib::~dynlib()
{
	if (handle == nullptr) return;

#ifdef _WIN32
	FreeLibrary((HMODULE) handle);
#else
	dlclose(handle);
#endif

}

void* rfkt::platform::dynlib::symbol_impl(const std::string& name)
{
	if (handle == nullptr) return nullptr;

#ifdef _WIN32
	return GetProcAddress((HMODULE) handle, name.c_str());
#else
	return dlsym(handle, name.c_str());
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
#ifdef _WIN32
	SetThreadDescription(GetCurrentThread(), name.c_str());
#endif
}
