#include <Shlwapi.h>
#include <thumbcache.h>
#include <new>

class Flam3ThumbProvider : public IInitializeWithStream, public IThumbnailProvider
{
public:
	Flam3ThumbProvider() : ref_count(1), stream(NULL) {}
	~Flam3ThumbProvider() { if (stream) stream->Release(); }

	// IUnknown
	IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
	{
		static const QITAB qit[] = {
			QITABENT(Flam3ThumbProvider, IInitializeWithStream),
			QITABENT(Flam3ThumbProvider, IThumbnailProvider),
			{ 0 },
		};
		return QISearch(this, qit, riid, ppv);
	}

	IFACEMETHODIMP_(ULONG) AddRef()
	{
		return InterlockedIncrement(&ref_count);
	}

	IFACEMETHODIMP_(ULONG) Release()
	{
		ULONG ref = InterlockedDecrement(&ref_count);
		if (ref == 0) delete this;
		return ref;
	}

	// IInitializeWithStream
	IFACEMETHODIMP Initialize(IStream* stream, DWORD grf_mode);

	// IThumbnailProvider
	IFACEMETHODIMP GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha);

private:

	long ref_count;
	IStream* stream;

};

HRESULT Flam3ThumbProvider_CreateInstance(REFIID riid, void** ppv) {
	Flam3ThumbProvider* provider = new (std::nothrow) Flam3ThumbProvider();
	if (!provider) return E_OUTOFMEMORY;
	
	auto hr = provider->QueryInterface(riid, ppv);
	provider->Release();
	return hr;
}

HRESULT Flam3ThumbProvider::Initialize(IStream* stream, DWORD grf_mode) {
	return this->stream ? E_UNEXPECTED : stream->QueryInterface(&this->stream);
}

HRESULT Flam3ThumbProvider::GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha) {
	return E_UNEXPECTED;
}