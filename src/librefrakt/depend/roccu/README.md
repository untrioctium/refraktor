# Roccu - Runtime selection of HIP/CUDA for C++
Roccu (_rock you_) is a C++ library for dynamically loading either CUDA or HIP GPU libraries at runtime. The API is patterned after CUDA and should function as a dropin replacement once symbol names are changed; `cu` symbols become `ru` and `CUDA_` macros become `RU_`. Only a small subset of CUDA functions needed for Refraktor are currently implemented.

## Installation
Roccu has no build dependencies, so it may be included directly in your project. If you are using CMake, you may include it through your preferred method (e.g. `add_subdirectory`, `FetchContent`, `ExternalProject`) and link against the `roccu` target.

## Hello World
```cpp
#include <roccu.h>

int main() {
    // initialize Roccu; selects an API and dynamically loads it
    auto api = roccuInit();

    // Make sure an API was selected
    if (api == ROCCU_API_NONE) { return 1; }

    // initialize the selected API and create a context
    ruInit(0);

    RUdevice device;
    ruDeviceGet(&device, 0);

    RUcontext context;
    ruCtxCreate(&context, 0, device);

    // do whatever you'd normally do with CUDA or HIP
    RUdeviceptr some_memory;
    ruMemAlloc(&some_memory, 1024);
    /* ... */

    return 0;
}
```

## API
Roccu's API is patterned after CUDA's API. The main difference is that all `cu` symbols are replaced with `ru` and all `CUDA_` macros are replaced with `RU_`. For example, `cuInit` becomes `ruInit` and `CUDA_SUCCESS` becomes `RU_SUCCESS`. If the CUDA API is selected, then these functions are calls directly into the dynamically loaded CUDA API. If HIP is selected, minimal shim functions are called that convert the HIP API into the CUDA API. 

Functions beginning with `roccu` or macros beginning with `ROCCU_` are part of the Roccu API itself.

A few Roccu specific functions are provided:
* `roccu_api roccuInit(int flags = 0)` - Initialize Roccu; if an API is selected, it the value of the `roccu_api` enum is returned. If no suitable API is found, `ROCCU_API_NONE` is returned; attempts to call `ru` functions without a valid API will return `ROCCU_ERROR_NOT_LOADED`. `flags` may be a bitwise combination of the following values:
    * `ROCCU_INIT_PREFER_HIP` - Prefer HIP over CUDA in machines that may have both GPU types. The default is to prefer CUDA.
    * `ROCCU_INIT_HOOK_ALLOCATION` - Hook memory allocation functions to track memory usage. The data structures tracking this use mutexes, so this may be a performance hit if many threads are allocating memory.
    * `ROCCU_INIT_NO_RTC` - Do not load runtime compiler functions; they will return `ROCCU_ERROR_NOT_LOADED` if called. Since HIP places its module linking functions in the RTC library, `ruLink*` APIs will be disabled in CUDA despite them being available; this is to ensure symmetry between the two APIs.
* `roccu_api roccuGetApi()` - Get the currently selected API.
* `const char* roccuGetApiName()` - Get the string name of the currently selected API.
* `size_t roccuGetMemoryUsage()` - Get the total GPU memory usage of the currently selected API. This is only valid if `ROCCU_INIT_HOOK_ALLOCATION` was passed to `roccuInit`; it will return 0 otherwise.
* `roccuPrintAllocations()` - Print a list of all allocations made by the currently selected API. This is only valid if `ROCCU_INIT_HOOK_ALLOCATION` was passed to `roccuInit`; it will do nothing otherwise. Stacktraces detailing each allocation are provided.

`roccu_vector_types.h` provides vector types familiar to CUDA/HIP (e.g. `int2`, `double4`, `dim3`). It does not depend on `roccu.h` and may be used independently.

## C++ API Wrappers
`roccu_cpp_types.h` provides some C++ wrapper types that may be useful. 
* `roccu::device_t` - Device handle wrapper that provides convienience functions for getting device information.  
* `roccu::context_t` - Context handle wrapper. `context_t::current()` returns the current context. Does not own the underlying context handle.
* `roccu::gpu_event` - Owning event handle wrapper.
* `roccu::gpu_stream` - Owning stream handle wrapper.
* `roccu::gpu_buffer<Contained>` - A device pointer wrapper that owns the memory it points to. `Contained` specifies the value type; if not provided, it defaults to `std::byte`. Provides various methods for copying to and from the buffer.
* `roccu::gpu_span<Contained>` - A device pointer wrapper that does not own the memory it points to; it is analogous to `std::span`. The API is equivalent to `gpu_buffer` but does not provide methods for allocating or freeing device memory. May be constructed from a `gpu_buffer` or a raw device pointer and size in elements.
* `roccu::l2_persister` (CUDA only) - RAII class that controls L2 persistence for a specified buffer. HIP does not provide an equivalent feature, so this class is a no-op in that case.

## Runtime compilation notes
While CUDA and HIP runtimes are provided by their respective GPU drivers, their runtime compilation libraries are not. You will need to provide them when distributing your application; the working directory will be searched for these libraries, and failing that, directories in the `PATH` environment variable are searched. If the libraries are not located, `roccuInit` will fail unless `ROCCU_INIT_NO_RTC` is passed. 

You will need to provide the following files for each API; they can be found in their respective platform SDKs. The first library in the list is what Roccu will search for and dynamically load, and the highest versioned library found will be selected.
* CUDA:
    * Windows: `nvrtc64*.dll`, `nvrtc-builtins64*.dll`
* HIP:
    * Windows: `hiprtc*.dll`, `hiprtc-builtins*.dll`, `amd_comgr*.dll`
