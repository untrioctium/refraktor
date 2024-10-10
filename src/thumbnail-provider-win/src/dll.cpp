#include <objbase.h>
#include <Shlwapi.h>
#include <thumbcache.h>
#include <ShlObj.h>
#include <new>
#include <cstdio>
#include <chrono>

#define CLSID_FLAM3THUMBHANDLER L"{892b95d4-3b68-4804-af11-46ac3e88b36b}"
#define FLAM3THUMBHANDLER L"Flam3ThumbnailHandler"
#define EXPORT comment(linker, "/EXPORT:" __FUNCTION__)

const CLSID CLSID_Flam3ThumbProvider = { 0x892b95d4, 0x3b68, 0x4804, { 0xaf, 0x11, 0x46, 0xac, 0x3e, 0x88, 0xb3, 0x6b } };

using PFNCREATEINSTANCE = HRESULT(*)(REFIID, void**);
struct CLASS_OBJECT_INIT
{
	CLSID const* pClsid;
	PFNCREATEINSTANCE pfnCreate;
};

HRESULT Flam3ThumbProvider_CreateInstance(REFIID riid, void **ppv);

const CLASS_OBJECT_INIT c_rgClassObjectInit[] =
{
	{ &CLSID_Flam3ThumbProvider, Flam3ThumbProvider_CreateInstance },
};

long dll_ref_count = 0;
HINSTANCE dll_instance = nullptr;

template<typename... Args>
void write_log(const char* format, Args... args)
{
    char buffer[1024];
    sprintf_s(buffer, format, args...);

    auto unix_timestamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    char buffer2[1024];
    sprintf_s(buffer2, "%d: (%d, %p) %s", unix_timestamp, GetCurrentProcessId(), dll_instance, buffer);

    // write to a log file
    FILE* f = nullptr;
    fopen_s(&f, "C:\\Users\\Public\\flam3.log", "a");
    if (f)
    {
        fprintf(f, "%s\n", buffer2);
        fclose(f);
    }
}

STDAPI_(BOOL) DllMain(HINSTANCE hInstance, DWORD dwReason, void*)
{
#pragma EXPORT

    constexpr auto reason_to_string = [](DWORD reason) -> const char*
	{
		switch (reason)
		{
		case DLL_PROCESS_ATTACH: return "DLL_PROCESS_ATTACH";
		case DLL_PROCESS_DETACH: return "DLL_PROCESS_DETACH";
		case DLL_THREAD_ATTACH: return "DLL_THREAD_ATTACH";
		case DLL_THREAD_DETACH: return "DLL_THREAD_DETACH";
		default: return "UNKNOWN";
		}
	};

	if (dwReason == DLL_PROCESS_ATTACH)
	{
		dll_instance = hInstance;
		DisableThreadLibraryCalls(hInstance);
	}

    write_log("DllMain: %s", reason_to_string(dwReason));

	return TRUE;
}

STDAPI DllCanUnloadNow()
{
#pragma EXPORT
	return dll_ref_count > 0 ? S_FALSE : S_OK;
}

void DllAddRef()
{
	InterlockedIncrement(&dll_ref_count);
}

void DllRelease()
{
	InterlockedDecrement(&dll_ref_count);
}

class CClassFactory : public IClassFactory
{
public:
    static HRESULT CreateInstance(REFCLSID clsid, const CLASS_OBJECT_INIT* pClassObjectInits, size_t cClassObjectInits, REFIID riid, void** ppv)
    {
        *ppv = NULL;
        HRESULT hr = CLASS_E_CLASSNOTAVAILABLE;
        for (size_t i = 0; i < cClassObjectInits; i++)
        {
            if (clsid == *pClassObjectInits[i].pClsid)
            {
                IClassFactory* pClassFactory = new (std::nothrow) CClassFactory(pClassObjectInits[i].pfnCreate);
                hr = pClassFactory ? S_OK : E_OUTOFMEMORY;
                if (SUCCEEDED(hr))
                {
                    hr = pClassFactory->QueryInterface(riid, ppv);
                    pClassFactory->Release();
                }
                break; // match found
            }
        }
        return hr;
    }

    CClassFactory(PFNCREATEINSTANCE pfnCreate) : _cRef(1), _pfnCreate(pfnCreate)
    {
        DllAddRef();
    }

    // IUnknown
    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
    {
        static const QITAB qit[] =
        {
            QITABENT(CClassFactory, IClassFactory),
            { 0 }
        };
        return QISearch(this, qit, riid, ppv);
    }

    IFACEMETHODIMP_(ULONG) AddRef()
    {
        return InterlockedIncrement(&_cRef);
    }

    IFACEMETHODIMP_(ULONG) Release()
    {
        long cRef = InterlockedDecrement(&_cRef);
        if (cRef == 0)
        {
            delete this;
        }
        return cRef;
    }

    // IClassFactory
    IFACEMETHODIMP CreateInstance(IUnknown* punkOuter, REFIID riid, void** ppv)
    {
        return punkOuter ? CLASS_E_NOAGGREGATION : _pfnCreate(riid, ppv);
    }

    IFACEMETHODIMP LockServer(BOOL fLock)
    {
        if (fLock)
        {
            DllAddRef();
        }
        else
        {
            DllRelease();
        }
        return S_OK;
    }

private:
    ~CClassFactory()
    {
        DllRelease();
    }

    long _cRef;
    PFNCREATEINSTANCE _pfnCreate;
};

STDAPI DllGetClassObject(REFCLSID clsid, REFIID riid, void** ppv)
{
#pragma EXPORT
    return CClassFactory::CreateInstance(clsid, c_rgClassObjectInit, ARRAYSIZE(c_rgClassObjectInit), riid, ppv);
}

// A struct to hold the information required for a registry entry

struct REGISTRY_ENTRY
{
    HKEY   hkeyRoot;
    PCWSTR pszKeyName;
    PCWSTR pszValueName;
    PCWSTR pszData;
};

// Creates a registry key (if needed) and sets the default value of the key

HRESULT CreateRegKeyAndSetValue(const REGISTRY_ENTRY* pRegistryEntry)
{
    HKEY hKey;
    HRESULT hr = HRESULT_FROM_WIN32(RegCreateKeyExW(pRegistryEntry->hkeyRoot, pRegistryEntry->pszKeyName,
        0, NULL, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &hKey, NULL));
    if (SUCCEEDED(hr))
    {
        hr = HRESULT_FROM_WIN32(RegSetValueExW(hKey, pRegistryEntry->pszValueName, 0, REG_SZ,
            (LPBYTE)pRegistryEntry->pszData,
            ((DWORD)wcslen(pRegistryEntry->pszData) + 1) * sizeof(WCHAR)));
        RegCloseKey(hKey);
    }
    return hr;
}

//
// Registers this COM server
//
STDAPI DllRegisterServer()
{
#pragma EXPORT

    write_log("DllRegisterServer");

    HRESULT hr;

    WCHAR szModuleName[MAX_PATH];

    if (!GetModuleFileNameW(dll_instance, szModuleName, ARRAYSIZE(szModuleName)))
    {
        hr = HRESULT_FROM_WIN32(GetLastError());
    }
    else
    {
        // List of registry entries we want to create
        const REGISTRY_ENTRY rgRegistryEntries[] =
        {
            // RootKey            KeyName                                                                ValueName                     Data
            {HKEY_CURRENT_USER,   L"Software\\Classes\\CLSID\\" CLSID_FLAM3THUMBHANDLER,                                 NULL,                           FLAM3THUMBHANDLER},
            {HKEY_CURRENT_USER,   L"Software\\Classes\\CLSID\\" CLSID_FLAM3THUMBHANDLER L"\\InProcServer32",             NULL,                           szModuleName},
            {HKEY_CURRENT_USER,   L"Software\\Classes\\CLSID\\" CLSID_FLAM3THUMBHANDLER L"\\InProcServer32",             L"ThreadingModel",              L"Apartment"},
            {HKEY_CURRENT_USER,   L"Software\\Classes\\.flam3\\ShellEx\\{e357fccd-a995-4576-b01f-234630154e96}",        NULL,                       CLSID_FLAM3THUMBHANDLER},
        };

        hr = S_OK;
        for (int i = 0; i < ARRAYSIZE(rgRegistryEntries) && SUCCEEDED(hr); i++)
        {
            hr = CreateRegKeyAndSetValue(&rgRegistryEntries[i]);
        }
    }
    if (SUCCEEDED(hr))
    {
        // This tells the shell to invalidate the thumbnail cache.  This is important because any .recipe files
        // viewed before registering this handler would otherwise show cached blank thumbnails.
        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, NULL, NULL);
    }
    return hr;
}

//
// Unregisters this COM server
//
STDAPI DllUnregisterServer()
{
#pragma EXPORT
    write_log("DllUnregisterServer");
    HRESULT hr = S_OK;

    const PCWSTR rgpszKeys[] =
    {
        L"Software\\Classes\\CLSID\\" CLSID_FLAM3THUMBHANDLER,
        L"Software\\Classes\\.flam3\\ShellEx\\{e357fccd-a995-4576-b01f-234630154e96}"
    };

    // Delete the registry entries
    for (int i = 0; i < ARRAYSIZE(rgpszKeys) && SUCCEEDED(hr); i++)
    {
        hr = HRESULT_FROM_WIN32(RegDeleteTreeW(HKEY_CURRENT_USER, rgpszKeys[i]));
        if (hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND))
        {
            // If the registry entry has already been deleted, say S_OK.
            hr = S_OK;
        }
    }
    return hr;
}