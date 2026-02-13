#pragma once

#include <cstddef>

using CUdevice = int;
using CUresult = int;
using nvrtcResult = int;
using CUdeviceptr = unsigned long long;
using CUhostFn = void (*)(void* userData);

constexpr static CUresult CUDA_SUCCESS = 0;
constexpr static CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
constexpr static nvrtcResult NVRTC_SUCCESS = 0;

#define ROCCU_DEFINE_OPAQUE(name) \
    using name = struct name##_st*

ROCCU_DEFINE_OPAQUE(CUcontext);
ROCCU_DEFINE_OPAQUE(CUstream);
ROCCU_DEFINE_OPAQUE(CUmodule);
ROCCU_DEFINE_OPAQUE(CUfunction);
ROCCU_DEFINE_OPAQUE(CUgraphicsResource);
ROCCU_DEFINE_OPAQUE(CUarray);
ROCCU_DEFINE_OPAQUE(CUlinkState);
ROCCU_DEFINE_OPAQUE(CUevent);
ROCCU_DEFINE_OPAQUE(nvrtcProgram);
ROCCU_DEFINE_OPAQUE(CUgreenCtx);
ROCCU_DEFINE_OPAQUE(CUdevResourceDesc);

enum CUjitInputType {
    CU_JIT_INPUT_CUBIN = 0,
    CU_JIT_INPUT_PTX,
    CU_JIT_INPUT_FATBINARY,
    CU_JIT_INPUT_OBJECT,
    CU_JIT_INPUT_LIBRARY
};

enum CUmemorytype {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4
};

enum CUfunction_attribute {
	CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
	CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
	CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
	CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
	CU_FUNC_ATTRIBUTE_NUM_REGS,
	CU_FUNC_ATTRIBUTE_PTX_VERSION,
	CU_FUNC_ATTRIBUTE_BINARY_VERSION,
	CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
	CU_FUNC_ATTRIBUTE_MAX
};

enum CUlimit {
    CU_LIMIT_STACK_SIZE = 0x00,
    CU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
    CU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06
};

enum CUlaunchAttributeID {
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
};

enum CUaccessProperty {
    CU_ACCESS_PROPERTY_NORMAL = 0,
    CU_ACCESS_PROPERTY_STREAMING = 1,
    CU_ACCESS_PROPERTY_PERSISTING = 2
};

constexpr static unsigned int CU_EVENT_DEFAULT = 0x0;
constexpr static unsigned int CU_EVENT_BLOCKING_SYNC = 0x1;
constexpr static unsigned int CU_EVENT_DISABLE_TIMING = 0x2;
constexpr static unsigned int CU_EVENT_INTERPROCESS = 0x4;

struct CUaccessPolicyWindow {
    CUdeviceptr basePtr;
    size_t numBytes;
    float hitRatio;
    CUaccessProperty hitProp;
    CUaccessProperty missProp;
};

union CUlaunchAttributeValue {
    char pad[64]; // NOLINT(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-avoid-magic-numbers)
    CUaccessPolicyWindow accessPolicyWindow;
};

struct CUDA_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;

    CUmemorytype srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;

    size_t dstXInBytes;
    size_t dstY;

    CUmemorytype dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;

    size_t WidthInBytes;
    size_t Height;
};

enum roccu_api {
    ROCCU_API_CUDA,
    ROCCU_API_ROCM,
    ROCCU_API_NONE
};

#ifdef ROCCU_IMPL

#include <unordered_map>

enum source_t {
    CU_DRIVER,
    CU_RTC
};

struct ru_traits {

    void** func;
    void** dll_sym;

    const char* name;
    source_t source;
    const char* cuda_name;
    const char* rocm_name;

    void* noop;
};

inline static std::unordered_map<const char*, ru_traits> ru_map = {};

template<typename T>
consteval static auto noop_ret(bool positive) {
    if constexpr (std::is_same_v<T, CUresult>) return positive ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
	else if constexpr(std::is_same_v<T, nvrtcResult>) return positive ? NVRTC_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
    else if constexpr (std::is_same_v<T, const char*>) return positive ? "" : "Roccu not initialized";
}

template<bool Positive, typename Ret, typename... Args>
consteval auto get_noop(Ret(*)(Args...)) {
	return [](Args...) -> Ret { 
        constexpr auto ret = noop_ret<Ret>(Positive);
        return ret; 
    };
}

template<typename FunctionPtrType>
bool register_traits(const ru_traits& traits) {

    ru_map[traits.name] = traits;
    ru_map[traits.name].noop = get_noop<true>((FunctionPtrType)nullptr);
    return true;
};

#endif

enum roccu_init_flags {
    ROCCU_INIT_NONE = 0,
    ROCCU_INIT_PREFER_HIP = 0x1,
    ROCCU_INIT_HOOK_ALLOCATION = 0x2,
    ROCCU_INIT_NO_RTC = 0x04
};

roccu_api roccuInit(int flags = ROCCU_INIT_NONE);
roccu_api roccuGetApi();
const char* roccuGetApiName();
size_t roccuGetMemoryUsage();
void roccuPrintAllocations();

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)

#ifndef ROCCU_IMPL
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) extern RET(*cu ## name)ARGS
    #define ROCCU_DEFINE_RTC_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) extern RET(*nv ## name)ARGS
#else 
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) \
        RET(*cu ## name)ARGS = get_noop<false>((RET(*)ARGS)nullptr); \
        RET(*cu ## name ## _dllsym)ARGS = nullptr; \
		static bool name##_init = register_traits<RET(*)ARGS>({(void**)& cu ## name, (void**)& cu ## name ## _dllsym, #name, SRC, #CUDA_NAME, #ROCM_NAME});

    #define ROCCU_DEFINE_RTC_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) \
        RET(*nv ## name)ARGS = get_noop<false>((RET(*)ARGS)nullptr); \
        RET(*nv ## name ## _dllsym)ARGS = nullptr; \
		static bool name##_init = register_traits<RET(*)ARGS>({(void**)& nv ## name, (void**)& nv ## name ## _dllsym, #name, SRC, #CUDA_NAME, #ROCM_NAME});
#endif

#define ROCCU_DEFINE_DIRECT_FUNC(name, SRC, RET, ARGS) ROCCU_DEFINE_FUNC(name, SRC, cu##name, hip##name, RET, ARGS)
#define ROCCU_DEFINE_DIRECT_RTC_FUNC(name, SRC, RET, ARGS) ROCCU_DEFINE_RTC_FUNC(name, SRC, nv##name, hip##name, RET, ARGS)
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

ROCCU_DEFINE_FUNC(CtxCreate, CU_DRIVER, cuCtxCreate_v2, hipCtxCreate, CUresult, (CUcontext* pctx, unsigned int flags, CUdevice dev));
ROCCU_DEFINE_FUNC(CtxDestroy, CU_DRIVER, cuCtxDestroy_v2, hipCtxDestroy, CUresult, (CUcontext ctx));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetCurrent, CU_DRIVER, CUresult, (CUcontext* pctx));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetDevice, CU_DRIVER, CUresult, (CUdevice* pdev));
ROCCU_DEFINE_FUNC(CtxGetStreamPriorityRange, CU_DRIVER, cuCtxGetStreamPriorityRange, hipDeviceGetStreamPriorityRange, CUresult, (int* leastPriority, int* greatestPriority));
ROCCU_DEFINE_DIRECT_FUNC(CtxSetCurrent, CU_DRIVER, CUresult, (CUcontext ctx));
ROCCU_DEFINE_FUNC(CtxSetLimit, CU_DRIVER, cuCtxSetLimit, NOOP, CUresult,(CUlimit limit, size_t value));

ROCCU_DEFINE_DIRECT_FUNC(DeviceGet, CU_DRIVER, CUresult, (CUdevice* device, int ordinal));

enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134
};

ROCCU_DEFINE_DIRECT_FUNC(DeviceGetAttribute, CU_DRIVER, CUresult, (int* pi, CUdevice_attribute attr, CUdevice dev));
ROCCU_DEFINE_DIRECT_FUNC(DeviceGetName, CU_DRIVER, CUresult, (char* name, int len, CUdevice dev));

ROCCU_DEFINE_DIRECT_FUNC(EventCreate, CU_DRIVER, CUresult, (CUevent* phEvent, unsigned int Flags));
ROCCU_DEFINE_FUNC(EventDestroy, CU_DRIVER, cuEventDestroy_v2, hipEventDestroy, CUresult, (CUevent phEvent));
ROCCU_DEFINE_DIRECT_FUNC(EventElapsedTime, CU_DRIVER, CUresult, (float* pMilliseconds, CUevent hStart, CUevent hEnd));
ROCCU_DEFINE_DIRECT_FUNC(EventQuery, CU_DRIVER, CUresult, (CUevent hEvent));
ROCCU_DEFINE_DIRECT_FUNC(EventRecord, CU_DRIVER, CUresult, (CUevent hEvent, CUstream hStream));
ROCCU_DEFINE_DIRECT_FUNC(EventSynchronize, CU_DRIVER, CUresult, (CUevent hEvent));

ROCCU_DEFINE_DIRECT_FUNC(FuncGetAttribute, CU_DRIVER, CUresult,(int* pi, CUfunction_attribute attrib, CUfunction hfunc));

ROCCU_DEFINE_DIRECT_FUNC(GetErrorString, CU_DRIVER, CUresult, (CUresult error, const char** pStr));
ROCCU_DEFINE_DIRECT_FUNC(GetErrorName, CU_DRIVER, CUresult, (CUresult error, const char** pStr));

ROCCU_DEFINE_DIRECT_FUNC(GraphicsGLRegisterImage, CU_DRIVER, CUresult, (CUgraphicsResource* cuResource, unsigned int image, unsigned int target, unsigned int flags));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsMapResources, CU_DRIVER, CUresult, (unsigned int count, CUgraphicsResource* resources, CUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsSubResourceGetMappedArray, CU_DRIVER, CUresult, (CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsUnmapResources, CU_DRIVER, CUresult, (unsigned int count, CUgraphicsResource* resources, CUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsUnregisterResource, CU_DRIVER, CUresult, (CUgraphicsResource resources));

ROCCU_DEFINE_DIRECT_FUNC(Init, CU_DRIVER, CUresult, (unsigned int Flags));

ROCCU_DEFINE_FUNC(LaunchCooperativeKernel, CU_DRIVER, cuLaunchCooperativeKernel, hipModuleLaunchCooperativeKernel, CUresult, (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream stream, void** kernelParams));
ROCCU_DEFINE_DIRECT_FUNC(LaunchHostFunc, CU_DRIVER, CUresult, (CUstream stream, CUhostFn func, void* userData));
ROCCU_DEFINE_FUNC(LaunchKernel, CU_DRIVER, cuLaunchKernel, hipModuleLaunchKernel, CUresult, (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream stream, void** kernelParams, void** extra));

ROCCU_DEFINE_FUNC(LinkAddData, CU_DRIVER, cuLinkAddData_v2, hiprtcLinkAddData, CUresult, (CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, const char** options, void** optionValues));
ROCCU_DEFINE_FUNC(LinkComplete, CU_DRIVER, cuLinkComplete, hiprtcLinkCreate, CUresult, (CUlinkState state, void** cubinOut, size_t* sizeOut));
ROCCU_DEFINE_FUNC(LinkCreate, CU_DRIVER, cuLinkCreate_v2, hiprtcLinkCreate, CUresult, (unsigned int numOptions, void** options, void** optionValues, CUlinkState* stateOut));
ROCCU_DEFINE_FUNC(LinkDestroy, CU_DRIVER, cuLinkDestroy, hiprtcLinkDestroy, CUresult, (CUlinkState state));

ROCCU_DEFINE_FUNC(MemAlloc, CU_DRIVER, cuMemAlloc_v2, hipMalloc, CUresult, (CUdeviceptr* dptr, size_t size));
ROCCU_DEFINE_FUNC(MemAllocAsync, CU_DRIVER, cuMemAllocAsync, hipMallocAsync, CUresult, (CUdeviceptr* dptr, size_t size, CUstream stream));
ROCCU_DEFINE_FUNC(MemAllocHost, CU_DRIVER, cuMemAllocHost_v2, hipHostMalloc, CUresult, (void** pp, size_t bytes));

ROCCU_DEFINE_FUNC(Memcpy2D, CU_DRIVER, cuMemcpy2D_v2, hipMemcpy2D, CUresult, (const CUDA_MEMCPY2D* pCopy));
ROCCU_DEFINE_FUNC(Memcpy2DAsync, CU_DRIVER, cuMemcpy2DAsync_v2, hipMemcpy2DAsync, CUresult, (const CUDA_MEMCPY2D* pCopy, CUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyDtoD, CU_DRIVER, cuMemcpyDtoD_v2, hipMemcpyDtoD, CUresult, (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyDtoDAsync, CU_DRIVER, cuMemcpyDtoDAsync_v2, hipMemcpyDtoDAsync, CUresult, (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyDtoH, CU_DRIVER, cuMemcpyDtoH_v2, hipMemcpyDtoH, CUresult, (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyDtoHAsync, CU_DRIVER, cuMemcpyDtoHAsync_v2, hipMemcpyDtoHAsync, CUresult, (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyHtoD, CU_DRIVER, cuMemcpyHtoD_v2, hipMemcpyHtoD, CUresult, (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyHtoDAsync, CU_DRIVER, cuMemcpyHtoDAsync_v2, hipMemcpyHtoDAsync, CUresult, (CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream));

ROCCU_DEFINE_FUNC(MemFree, CU_DRIVER, cuMemFree_v2, hipFree, CUresult, (CUdeviceptr dptr));
ROCCU_DEFINE_FUNC(MemFreeAsync, CU_DRIVER, cuMemFreeAsync, hipFreeAsync, CUresult, (CUdeviceptr dptr, CUstream stream));
ROCCU_DEFINE_FUNC(MemFreeHost, CU_DRIVER, cuMemFreeHost, hipHostFree, CUresult, (void* p));

ROCCU_DEFINE_FUNC(MemsetD8, CU_DRIVER, cuMemsetD8_v2, hipMemsetD8, CUresult, (CUdeviceptr dstDevice, unsigned char uc, size_t N));
ROCCU_DEFINE_DIRECT_FUNC(MemsetD8Async, CU_DRIVER, CUresult, (CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream stream));
ROCCU_DEFINE_FUNC(MemsetD16, CU_DRIVER, cuMemsetD16_v2, hipMemsetD16, CUresult, (CUdeviceptr dstDevice, unsigned short us, size_t N));
ROCCU_DEFINE_DIRECT_FUNC(MemsetD16Async, CU_DRIVER, CUresult, (CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream stream));
ROCCU_DEFINE_FUNC(MemsetD32, CU_DRIVER, cuMemsetD32_v2, hipMemsetD32, CUresult, (CUdeviceptr dstDevice, unsigned int ui, size_t N));
ROCCU_DEFINE_DIRECT_FUNC(MemsetD32Async, CU_DRIVER, CUresult, (CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream stream));

ROCCU_DEFINE_DIRECT_FUNC(ModuleGetFunction, CU_DRIVER, CUresult, (CUfunction* hfunc, CUmodule hmod, const char* name));
ROCCU_DEFINE_FUNC(ModuleGetGlobal, CU_DRIVER, cuModuleGetGlobal_v2, hipModuleGetGlobal, CUresult, (CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name));
ROCCU_DEFINE_FUNC(ModuleLoadDataEx, CU_DRIVER, cuModuleLoadDataEx, hipModuleLoadData, CUresult, (CUmodule* module, const void* image, unsigned int numOptions, const char** options, void** optionValues));
ROCCU_DEFINE_FUNC(ModuleLoadData, CU_DRIVER, cuModuleLoadData, hipModuleLoadData, CUresult, (CUmodule* module, const void* image));
ROCCU_DEFINE_DIRECT_FUNC(ModuleUnload, CU_DRIVER, CUresult, (CUmodule hmod));

ROCCU_DEFINE_FUNC(OccupancyMaxActiveBlocksPerMultiprocessor, CU_DRIVER, cuOccupancyMaxActiveBlocksPerMultiprocessor, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, CUresult, (int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize));
ROCCU_DEFINE_FUNC(OccupancyMaxPotentialBlockSize, CU_DRIVER, cuOccupancyMaxPotentialBlockSize, hipModuleOccupancyMaxPotentialBlockSize, CUresult, (int* minGridSize, int* blockSize, CUfunction func, void* blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit));

ROCCU_DEFINE_DIRECT_FUNC(StreamCreateWithPriority, CU_DRIVER, CUresult, (CUstream* pStream, unsigned int flags, int priority));
ROCCU_DEFINE_FUNC(StreamDestroy, CU_DRIVER, cuStreamDestroy_v2, hipStreamDestroy, CUresult, (CUstream stream));
ROCCU_DEFINE_FUNC(StreamSetAttribute, CU_DRIVER, cuStreamSetAttribute, NOOP, CUresult, (CUstream stream, CUlaunchAttributeID attr, CUlaunchAttributeValue* value));
ROCCU_DEFINE_DIRECT_FUNC(StreamSynchronize, CU_DRIVER, CUresult, (CUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(StreamWaitEvent, CU_DRIVER, CUresult, (CUstream stream, CUevent event, unsigned int flags));
ROCCU_DEFINE_DIRECT_FUNC(StreamQuery, CU_DRIVER, CUresult, (CUstream stream));

ROCCU_DEFINE_RTC_FUNC(rtcAddNameExpression, CU_RTC, nvrtcAddNameExpression, hiprtcAddNameExpression, nvrtcResult, (nvrtcProgram prog, const char* name_expression));
ROCCU_DEFINE_RTC_FUNC(rtcCompileProgram, CU_RTC, nvrtcCompileProgram, hiprtcCompileProgram, nvrtcResult, (nvrtcProgram prog, int numOptions, const char** options));
ROCCU_DEFINE_RTC_FUNC(rtcCreateProgram, CU_RTC, nvrtcCreateProgram, hiprtcCreateProgram, nvrtcResult, (nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames));
ROCCU_DEFINE_RTC_FUNC(rtcDestroyProgram, CU_RTC, nvrtcDestroyProgram, hiprtcDestroyProgram, nvrtcResult, (nvrtcProgram* prog));
ROCCU_DEFINE_RTC_FUNC(rtcGetErrorString, CU_RTC, nvrtcGetErrorString, hiprtcGetErrorString, const char*, (nvrtcResult));
ROCCU_DEFINE_RTC_FUNC(rtcGetLoweredName, CU_RTC, nvrtcGetLoweredName, hiprtcGetLoweredName, nvrtcResult, (nvrtcProgram prog, const char* name_expression, const char** lowered_name));
ROCCU_DEFINE_RTC_FUNC(rtcGetProgramLog, CU_RTC, nvrtcGetProgramLog, hiprtcGetProgramLog, nvrtcResult, (nvrtcProgram prog, char* log));
ROCCU_DEFINE_RTC_FUNC(rtcGetProgramLogSize, CU_RTC, nvrtcGetProgramLogSize, hiprtcGetProgramLogSize, nvrtcResult, (nvrtcProgram prog, size_t* logSizeRet));
ROCCU_DEFINE_RTC_FUNC(rtcGetCUBIN, CU_RTC, nvrtcGetCUBIN, hiprtcGetCode, nvrtcResult, (nvrtcProgram prog, char* ptx));
ROCCU_DEFINE_RTC_FUNC(rtcGetCUBINSize, CU_RTC, nvrtcGetCUBINSize, hiprtcGetCodeSize, nvrtcResult, (nvrtcProgram prog, size_t* ptxSizeRet));

#undef ROCCU_DEFINE_FUNC
#undef ROCCU_DEFINE_DIRECT_FUNC
#undef ROCCU_DEFINE_OPAQUE
#undef ROCCU_DEFINE_RTC_FUNC
#undef ROCCU_DEFINE_DIRECT_RTC_FUNC

#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#define ROCCU_HAS_EXCEPTIONS 1
#else
#define ROCCU_HAS_EXCEPTIONS 0
#endif


#if __cpp_lib_stacktrace >= 202002L
#include <stacktrace>
#define ROCCU_HAS_STACKTRACE 1
#else
#define ROCCU_HAS_STACKTRACE 0
#endif

#ifdef ROCCU_HAS_EXCEPTIONS
#include <stdexcept>
namespace roccu {
    class cuda_error : public std::runtime_error {
    public:
        cuda_error(CUresult result, const char* description = nullptr) : std::runtime_error(make_error_string(result, description)), result_(result) {}

        CUresult code() const { return result_; }

    private:

        static std::string make_error_string(CUresult result, const char* description = nullptr) {
            std::string error_string;

            const char* cuda_name = nullptr;
            cuGetErrorName(result, &cuda_name);
            if(cuda_name) error_string += cuda_name;
            else error_string += "Unknown error";

            if(!description) return error_string;
            error_string += " caused by '";
            error_string += description;
            error_string += "'";
            return error_string;
        }

        CUresult result_;
    };

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    #define ROCCU_SAFE_CALL(x) do { if(auto result = x; result != CUDA_SUCCESS) throw roccu::cuda_error(result, #x); } while(0)

}
#endif