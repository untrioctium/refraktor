#include <optional>
#include <string_view>
#include <cstdio>
#include <array>
#include <atomic>
#include <span>
#include <mutex>
#include <variant>

#include <stacktrace>
#include <filesystem>

namespace fs = std::filesystem;

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#define ROCCU_IMPL
#include "roccu.h"

struct roccu_load_info {
    roccu_api api;
    const char* driver;
    const char* rtc;
};


#ifdef _WIN32
inline const static auto roccu_load_info_list = std::array{
    roccu_load_info{ROCCU_API_CUDA, "nvcuda.dll", "nvrtc64_120_0.dll"},
    roccu_load_info{ROCCU_API_ROCM, "amdhip64.dll", "hiprtc0507.dll"}
};
#endif


class dynamic_library {
public:

    static std::optional<dynamic_library> load(const char* name) {

#ifdef _WIN32
        auto handle = LoadLibraryA(name);
        if (!handle) return std::nullopt;
        return dynamic_library{handle};
#else
        auto handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
		if (!handle) return std::nullopt;
		return dynamic_library{handle};
#endif

    }

    dynamic_library(const dynamic_library&) = delete;
    dynamic_library& operator=(const dynamic_library&) = delete;

    dynamic_library(dynamic_library&& o) noexcept {
        std::swap(handle, o.handle);
    }

    dynamic_library& operator=(dynamic_library&& o) noexcept {
		std::swap(handle, o.handle);
		return *this;
	}

    ~dynamic_library() {
#ifdef _WIN32
        if(handle) FreeLibrary(handle);
#else
        if(handle) dlclose(handle);
#endif
    }
    
    void* function(const char* name) const {
#ifdef _WIN32
        return GetProcAddress(handle, name);
#else
        return dlsym(handle, name);
#endif
    }

private:

    using handle_t = decltype([]() {
#ifdef _WIN32
		return (HMODULE)nullptr;
#else
        return (void*)nullptr;
#endif
    }());

    explicit dynamic_library(handle_t handle) : handle(handle) {}

    handle_t handle = nullptr;
};

struct roccu_impl {
	roccu_api api;
    dynamic_library driver_lib;
    std::optional<dynamic_library> rtc_lib;
};

static inline std::optional<roccu_impl> api;
constinit inline static std::atomic_size_t mem_alloc_size = 0;

inline static std::unordered_map<RUdeviceptr, std::pair<size_t, std::stacktrace>> mem_alloc_map = {};
inline static std::mutex mem_alloc_mutex = {};

enum hipDeviceAttribute_t {
    hipDeviceAttributeCudaCompatibleBegin = 0,

    hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin, ///< Whether ECC support is enabled.
    hipDeviceAttributeAccessPolicyMaxWindowSize,        ///< Cuda only. The maximum size of the window policy in bytes.
    hipDeviceAttributeAsyncEngineCount,                 ///< Cuda only. Asynchronous engines number.
    hipDeviceAttributeCanMapHostMemory,                 ///< Whether host memory can be mapped into device address space
    hipDeviceAttributeCanUseHostPointerForRegisteredMem,///< Cuda only. Device can access host registered memory
    ///< at the same virtual address as the CPU
    hipDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeComputeMode,                      ///< Compute mode that device is currently in.
    hipDeviceAttributeComputePreemptionSupported,       ///< Cuda only. Device supports Compute Preemption.
    hipDeviceAttributeConcurrentKernels,                ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributeConcurrentManagedAccess,          ///< Device can coherently access managed memory concurrently with the CPU
    hipDeviceAttributeCooperativeLaunch,                ///< Support cooperative launch
    hipDeviceAttributeCooperativeMultiDeviceLaunch,     ///< Support cooperative launch on multiple devices
    hipDeviceAttributeDeviceOverlap,                    ///< Cuda only. Device can concurrently copy memory and execute a kernel.
    ///< Deprecated. Use instead asyncEngineCount.
    hipDeviceAttributeDirectManagedMemAccessFromHost,   ///< Host can directly access managed memory on
    ///< the device without migration
    hipDeviceAttributeGlobalL1CacheSupported,           ///< Cuda only. Device supports caching globals in L1
    hipDeviceAttributeHostNativeAtomicSupported,        ///< Cuda only. Link between the device and the host supports native atomic operations
    hipDeviceAttributeIntegrated,                       ///< Device is integrated GPU
    hipDeviceAttributeIsMultiGpuBoard,                  ///< Multiple GPU devices.
    hipDeviceAttributeKernelExecTimeout,                ///< Run time limit for kernels executed on the device
    hipDeviceAttributeL2CacheSize,                      ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeLocalL1CacheSupported,            ///< caching locals in L1 is supported
    hipDeviceAttributeLuid,                             ///< Cuda only. 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeLuidDeviceNodeMask,               ///< Cuda only. Luid device node mask. Undefined on TCC and non-Windows platforms
    hipDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    hipDeviceAttributeManagedMemory,                    ///< Device supports allocating managed memory on this system
    hipDeviceAttributeMaxBlocksPerMultiProcessor,       ///< Cuda only. Max block size per multiprocessor
    hipDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    hipDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    hipDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    hipDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    hipDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    hipDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    hipDeviceAttributeMaxSurface1D,                     ///< Maximum size of 1D surface.
    hipDeviceAttributeMaxSurface1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered surface.
    hipDeviceAttributeMaxSurface2D,                     ///< Maximum dimension (width, height) of 2D surface.
    hipDeviceAttributeMaxSurface2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered surface.
    hipDeviceAttributeMaxSurface3D,                     ///< Maximum dimension (width, height, depth) of 3D surface.
    hipDeviceAttributeMaxSurfaceCubemap,                ///< Cuda only. Maximum dimensions of Cubemap surface.
    hipDeviceAttributeMaxSurfaceCubemapLayered,         ///< Cuda only. Maximum dimension of Cubemap layered surface.
    hipDeviceAttributeMaxTexture1DWidth,                ///< Maximum size of 1D texture.
    hipDeviceAttributeMaxTexture1DLayered,              ///< Cuda only. Maximum dimensions of 1D layered texture.
    hipDeviceAttributeMaxTexture1DLinear,               ///< Maximum number of elements allocatable in a 1D linear texture.
    ///< Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
    hipDeviceAttributeMaxTexture1DMipmap,               ///< Cuda only. Maximum size of 1D mipmapped texture.
    hipDeviceAttributeMaxTexture2DWidth,                ///< Maximum dimension width of 2D texture.
    hipDeviceAttributeMaxTexture2DHeight,               ///< Maximum dimension hight of 2D texture.
    hipDeviceAttributeMaxTexture2DGather,               ///< Cuda only. Maximum dimensions of 2D texture if gather operations  performed.
    hipDeviceAttributeMaxTexture2DLayered,              ///< Cuda only. Maximum dimensions of 2D layered texture.
    hipDeviceAttributeMaxTexture2DLinear,               ///< Cuda only. Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
    hipDeviceAttributeMaxTexture2DMipmap,               ///< Cuda only. Maximum dimensions of 2D mipmapped texture.
    hipDeviceAttributeMaxTexture3DWidth,                ///< Maximum dimension width of 3D texture.
    hipDeviceAttributeMaxTexture3DHeight,               ///< Maximum dimension height of 3D texture.
    hipDeviceAttributeMaxTexture3DDepth,                ///< Maximum dimension depth of 3D texture.
    hipDeviceAttributeMaxTexture3DAlt,                  ///< Cuda only. Maximum dimensions of alternate 3D texture.
    hipDeviceAttributeMaxTextureCubemap,                ///< Cuda only. Maximum dimensions of Cubemap texture
    hipDeviceAttributeMaxTextureCubemapLayered,         ///< Cuda only. Maximum dimensions of Cubemap layered texture.
    hipDeviceAttributeMaxThreadsDim,                    ///< Maximum dimension of a block
    hipDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,      ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeMaxPitch,                         ///< Maximum pitch in bytes allowed by memory copies
    hipDeviceAttributeMemoryBusWidth,                   ///< Global memory bus width in bits.
    hipDeviceAttributeMemoryClockRate,                  ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    hipDeviceAttributeMultiGpuBoardGroupID,             ///< Cuda only. Unique ID of device group on the same multi-GPU board
    hipDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    hipDeviceAttributeName,                             ///< Device name.
    hipDeviceAttributePageableMemoryAccess,             ///< Device supports coherently accessing pageable memory
    ///< without calling hipHostRegister on it
    hipDeviceAttributePageableMemoryAccessUsesHostPageTables, ///< Device accesses pageable memory via the host's page tables
    hipDeviceAttributePciBusId,                         ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                      ///< PCI Device ID.
    hipDeviceAttributePciDomainID,                      ///< PCI Domain ID.
    hipDeviceAttributePersistingL2CacheMaxSize,         ///< Cuda11 only. Maximum l2 persisting lines capacity in bytes
    hipDeviceAttributeMaxRegistersPerBlock,             ///< 32-bit registers available to a thread block. This number is shared
    ///< by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeMaxRegistersPerMultiprocessor,    ///< 32-bit registers available per block.
    hipDeviceAttributeReservedSharedMemPerBlock,        ///< Cuda11 only. Shared memory reserved by CUDA driver per block.
    hipDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeSharedMemPerBlockOptin,           ///< Cuda only. Maximum shared memory per block usable by special opt in.
    hipDeviceAttributeSharedMemPerMultiprocessor,       ///< Cuda only. Shared memory available per multiprocessor.
    hipDeviceAttributeSingleToDoublePrecisionPerfRatio, ///< Cuda only. Performance ratio of single precision to double precision.
    hipDeviceAttributeStreamPrioritiesSupported,        ///< Cuda only. Whether to support stream priorities.
    hipDeviceAttributeSurfaceAlignment,                 ///< Cuda only. Alignment requirement for surfaces
    hipDeviceAttributeTccDriver,                        ///< Cuda only. Whether device is a Tesla device using TCC driver
    hipDeviceAttributeTextureAlignment,                 ///< Alignment requirement for textures
    hipDeviceAttributeTexturePitchAlignment,            ///< Pitch alignment requirement for 2D texture references bound to pitched memory;
    hipDeviceAttributeTotalConstantMemory,              ///< Constant memory size in bytes.
    hipDeviceAttributeTotalGlobalMem,                   ///< Global memory available on devicice.
    hipDeviceAttributeUnifiedAddressing,                ///< Cuda only. An unified address space shared with the host.
    hipDeviceAttributeUuid,                             ///< Cuda only. Unique ID in 16 byte.
    hipDeviceAttributeWarpSize,                         ///< Warp size in threads.
    hipDeviceAttributeMemoryPoolsSupported,             ///< Device supports HIP Stream Ordered Memory Allocator
    hipDeviceAttributeVirtualMemoryManagementSupported, ///< Device supports HIP virtual memory management

    hipDeviceAttributeCudaCompatibleEnd = 9999,
    hipDeviceAttributeAmdSpecificBegin = 10000,

    hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,  ///< Frequency in khz of the timer used by the device-side "clock*"
    hipDeviceAttributeArch,                                     ///< Device architecture
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,         ///< Maximum Shared Memory PerMultiprocessor.
    hipDeviceAttributeGcnArch,                                  ///< Device gcn architecture
    hipDeviceAttributeGcnArchName,                              ///< Device gcnArch name in 256 bytes
    hipDeviceAttributeHdpMemFlushCntl,                          ///< Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeHdpRegFlushCntl,                          ///< Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,      ///< Supports cooperative launch on multiple
    ///< devices with unmatched functions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,   ///< Supports cooperative launch on multiple
    ///< devices with unmatched grid dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,  ///< Supports cooperative launch on multiple
    ///< devices with unmatched block dimensions
    hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem, ///< Supports cooperative launch on multiple
    ///< devices with unmatched shared memories
    hipDeviceAttributeIsLargeBar,                               ///< Whether it is LargeBar
    hipDeviceAttributeAsicRevision,                             ///< Revision of the GPU in this device
    hipDeviceAttributeCanUseStreamWaitValue,                    ///< '1' if Device supports hipStreamWaitValue32() and
    ///< hipStreamWaitValue64(), '0' otherwise.
    hipDeviceAttributeImageSupport,                             ///< '1' if Device supports image, '0' otherwise.
    hipDeviceAttributePhysicalMultiProcessorCount,              ///< All available physical compute
    ///< units for the device
    hipDeviceAttributeFineGrainSupport,                         ///< '1' if Device supports fine grain, '0' otherwise
    hipDeviceAttributeWallClockRate,                            ///< Constant frequency of wall clock in kilohertz.

    hipDeviceAttributeAmdSpecificEnd = 19999,
    hipDeviceAttributeVendorSpecificBegin = 20000,
    // Extended attributes for vendors
};

using attr_conv_func = std::add_pointer_t<int(RUdevice dev)>;
using attrib_conv = std::variant<hipDeviceAttribute_t, int, attr_conv_func>;

const static std::unordered_map<RUdevice_attribute, attrib_conv> ru_to_hip_device_attribute = {
    {RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, hipDeviceAttributeComputeCapabilityMajor},
    {RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, hipDeviceAttributeComputeCapabilityMinor},
    {RU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, 0},
    {RU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, 0},
    {RU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, 16},
    {RU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, hipDeviceAttributeMaxThreadsPerMultiProcessor},
    {RU_DEVICE_ATTRIBUTE_WARP_SIZE, hipDeviceAttributeWarpSize},
    {RU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, hipDeviceAttributeMaxThreadsPerBlock},
    {RU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, hipDeviceAttributeMultiprocessorCount},
    {RU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor},
};

bool load_symbols(int flags) {
	if (!api) return false;

	bool all_found = true;

	for(auto& [name, traits] : ru_map) {
        bool force_rtc = api->api == ROCCU_API_ROCM && std::string_view{ traits.name }.starts_with("Link");
		const auto& lib = traits.source == RU_DRIVER && !force_rtc? api->driver_lib : api->rtc_lib.value();
		auto symbol_name = api->api == ROCCU_API_CUDA ? traits.cuda_name : traits.rocm_name;

		if (symbol_name == "NOOP") {
			*traits.func = traits.noop;
			printf("Warning: %s is not implemented for this API\n", traits.name);
			continue;
		}

        if(auto symbol = lib.function(symbol_name); symbol) {
			*traits.dll_sym = symbol;
			*traits.func = *traits.dll_sym;
		}
		else {
            printf("Failed to load symbol %s -> %s\n", traits.name, symbol_name);
			all_found = false;
		}

	}

    if(!all_found) return false;

	if (api->api == ROCCU_API_ROCM) {

		ruGetErrorString = +[](RUresult result, const char** ret) {
			auto original_func = reinterpret_cast<std::add_pointer_t<const char* (RUresult)>>(ruGetErrorString_dllsym);
			*ret = original_func(result);
			return RU_SUCCESS;
		};

        ruGetErrorName = +[](RUresult result, const char** ret) {
            auto original_func = reinterpret_cast<std::add_pointer_t<const char* (RUresult)>>(ruGetErrorName_dllsym);
            *ret = original_func(result);
            return RU_SUCCESS;
        };

        ruDeviceGetAttribute = +[](int* value, RUdevice_attribute attr, int device) -> RUresult {
			auto lut = ru_to_hip_device_attribute.find(attr);
            if(lut == ru_to_hip_device_attribute.end()) {
                __debugbreak();
			}

            if(std::holds_alternative<int>(lut->second)) {
				*value = std::get<int>(lut->second);
				return RU_SUCCESS;
            }
            else if (std::holds_alternative<attr_conv_func>(lut->second)) {
                *value = std::get<attr_conv_func>(lut->second)(device);
                return RU_SUCCESS;
            }
            else {
                auto hip_attr = std::bit_cast<RUdevice_attribute>(std::get<hipDeviceAttribute_t>(lut->second));
                return ruDeviceGetAttribute_dllsym(value, hip_attr, device);
            }
		};

        ruMemAllocHost = +[](void** ptr, size_t size) -> RUresult {
			auto original_func = reinterpret_cast<std::add_pointer_t<RUresult(void**, size_t, int)>>(ruMemAllocHost_dllsym);
			return original_func(ptr, size, 0);
		};
	
	}

	return all_found;
}

static void hook_allocation() {
	ruMemAlloc = [](RUdeviceptr* ptr, size_t size) {
		std::scoped_lock lock(mem_alloc_mutex);
		mem_alloc_size += size;
		auto ret = ruMemAlloc_dllsym(ptr, size);
		mem_alloc_map[*ptr] = { size, std::stacktrace::current() };
		return ret;
		};

	ruMemFree = [](RUdeviceptr ptr) {
		std::scoped_lock lock(mem_alloc_mutex);
		mem_alloc_size -= mem_alloc_map[ptr].first;
		mem_alloc_map.erase(ptr);
		return ruMemFree_dllsym(ptr);
	};

	ruMemAllocAsync = [](RUdeviceptr* ptr, size_t size, RUstream stream) {
		std::scoped_lock lock(mem_alloc_mutex);

		auto ret = ruMemAllocAsync_dllsym(ptr, size, stream);
		mem_alloc_map[*ptr] = { size, std::stacktrace::current() };

		ruLaunchHostFunc(stream, [](void* ptr) {
			mem_alloc_size += mem_alloc_map[reinterpret_cast<RUdeviceptr>(ptr)].first;
		}, reinterpret_cast<void*>(*ptr));

		return ret;
	};

	ruMemFreeAsync = [](RUdeviceptr ptr, RUstream stream) {
		auto ret = ruMemFreeAsync_dllsym(ptr, stream);

		ruLaunchHostFunc(stream, [](void* ptr) {
			std::scoped_lock lock(mem_alloc_mutex);
			RUdeviceptr dptr = reinterpret_cast<RUdeviceptr>(ptr);
			mem_alloc_size -= mem_alloc_map[dptr].first;
			mem_alloc_map.erase(dptr);

		}, reinterpret_cast<void*>(ptr));

		return ret;
	};
}

roccu_api roccuInit(int flags) {

	if(api) return api->api;

	api = []() -> std::optional<roccu_impl> {
		for(const auto& info : roccu_load_info_list) {
			auto driver = dynamic_library::load(info.driver);
            auto rtc = dynamic_library::load(info.rtc);

            if(driver && rtc) {
                return roccu_impl{info.api, std::move(driver.value()), std::move(rtc.value())};
			}
		}
		return std::nullopt;
	}();

	if(api) {
		if (load_symbols(flags)) {
			if(flags & ROCCU_INIT_HOOK_ALLOCATION) hook_allocation();
			return api->api;
		}
		else {
			api.reset();
		}
	}

	return ROCCU_API_NONE;
}

roccu_api roccuGetApi()
{
	return api? api->api : ROCCU_API_NONE;
}

const char* roccuGetApiName()
{
    switch(roccuGetApi()) {
		case ROCCU_API_CUDA: return "CUDA";
		case ROCCU_API_ROCM: return "ROCm";
		default: return "None";
	}
}

size_t roccuGetMemoryUsage() {
	return mem_alloc_size;
}

void roccuPrintAllocations() {
	std::scoped_lock lock(mem_alloc_mutex);
	for(const auto& [ptr, alloc] : mem_alloc_map) {
		printf("Allocation at %lluu of size %zu (%.1f MB)\n", ptr, alloc.first, alloc.first / (1024.0 * 1024.0));
		for(int i = 1; i < alloc.second.size(); i++) {
			auto& entry = alloc.second[i];
			printf("%s %d %s\n", entry.source_file().c_str(), entry.source_line(), entry.description().c_str());
		}
	}
}