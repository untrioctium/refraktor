#pragma once

#include <type_traits>
#include <roccu_vector_types.h>

using RUdevice = int;
using RUresult = int;
using rurtcResult = int;
using RUdeviceptr = unsigned long long;
using RUhostFn = void (*)(void* userData);

constexpr static RUresult RU_SUCCESS = 0;
constexpr static RUresult RU_ERROR_NOT_INITIALIZED = 3;
constexpr static rurtcResult RURTC_SUCCESS = 0;

#define ROCCU_DEFINE_OPAQUE(name) \
    using name = struct name##_st*

ROCCU_DEFINE_OPAQUE(RUcontext);
ROCCU_DEFINE_OPAQUE(RUstream);
ROCCU_DEFINE_OPAQUE(RUmodule);
ROCCU_DEFINE_OPAQUE(RUfunction);
ROCCU_DEFINE_OPAQUE(RUgraphicsResource);
ROCCU_DEFINE_OPAQUE(RUarray);
ROCCU_DEFINE_OPAQUE(RUlinkState);
ROCCU_DEFINE_OPAQUE(RUevent);
ROCCU_DEFINE_OPAQUE(rurtcProgram);

enum RUjitInputType {
    RU_JIT_INPUT_CUBIN = 0,
    RU_JIT_INPUT_PTX,
    RU_JIT_INPUT_FATBINARY,
    RU_JIT_INPUT_OBJECT,
    RU_JIT_INPUT_LIBRARY
};

enum RUmemorytype {
    RU_MEMORYTYPE_HOST = 1,
    RU_MEMORYTYPE_DEVICE = 2,
    RU_MEMORYTYPE_ARRAY = 3,
    RU_MEMORYTYPE_UNIFIED = 4
};

enum RUfunction_attribute {
	RU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
	RU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
	RU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
	RU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
	RU_FUNC_ATTRIBUTE_NUM_REGS,
	RU_FUNC_ATTRIBUTE_PTX_VERSION,
	RU_FUNC_ATTRIBUTE_BINARY_VERSION,
	RU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
	RU_FUNC_ATTRIBUTE_MAX
};

enum RUlimit {
    RU_LIMIT_STACK_SIZE = 0x00,
    RU_LIMIT_PRINTF_FIFO_SIZE = 0x01,
    RU_LIMIT_MALLOC_HEAP_SIZE = 0x02,
    RU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03,
    RU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    RU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
    RU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06
};

enum RUlaunchAttributeID {
    RU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
};

enum RUaccessProperty {
    RU_ACCESS_PROPERTY_NORMAL = 0,
    RU_ACCESS_PROPERTY_STREAMING = 1,
    RU_ACCESS_PROPERTY_PERSISTING = 2
};

constexpr static unsigned int RU_EVENT_DEFAULT = 0x0;
constexpr static unsigned int RU_EVENT_BLOCKING_SYNC = 0x1;
constexpr static unsigned int RU_EVENT_DISABLE_TIMING = 0x2;
constexpr static unsigned int RU_EVENT_INTERPROCESS = 0x4;

struct RUaccessPolicyWindow {
    RUdeviceptr basePtr;
    size_t numBytes;
    float hitRatio;
    RUaccessProperty hitProp;
    RUaccessProperty missProp;
};

using RUlaunchAttributeValue = union {
    char pad[64];
    RUaccessPolicyWindow accessPolicyWindow;
};

struct RU_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;

    RUmemorytype srcMemoryType;
    const void* srcHost;
    RUdeviceptr srcDevice;
    RUarray srcArray;
    size_t srcPitch;

    size_t dstXInBytes;
    size_t dstY;

    RUmemorytype dstMemoryType;
    void* dstHost;
    RUdeviceptr dstDevice;
    RUarray dstArray;
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
    RU_DRIVER,
    RU_RTC
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
    if constexpr (std::is_same_v<T, RUresult>) return positive ? RU_SUCCESS : RU_ERROR_NOT_INITIALIZED;
	else if constexpr(std::is_same_v<T, rurtcResult>) return positive ? RURTC_SUCCESS : RU_ERROR_NOT_INITIALIZED;
    else if constexpr (std::is_same_v<T, const char*>) return positive ? "" : "Roccu not initialized";
}

template<bool Positive, typename Ret, typename... Args>
consteval auto get_noop(Ret(*)(Args...)) {
	return [](Args...) -> Ret { return noop_ret<Ret>(Positive); };
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

#ifndef ROCCU_IMPL
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) extern RET(*ru ## name)ARGS
#else 
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) \
        RET(*ru ## name)ARGS = get_noop<false>((RET(*)ARGS)nullptr); \
        RET(*ru ## name ## _dllsym)ARGS = nullptr; \
		static bool name##_init = register_traits<RET(*)ARGS>({(void**)& ru ## name, (void**)& ru ## name ## _dllsym, #name, SRC, #CUDA_NAME, #ROCM_NAME});
#endif

#define ROCCU_DEFINE_DIRECT_FUNC(name, SRC, RET, ARGS) ROCCU_DEFINE_FUNC(name, SRC, cu##name, hip##name, RET, ARGS)

ROCCU_DEFINE_FUNC(CtxCreate, RU_DRIVER, cuCtxCreate_v2, hipCtxCreate, RUresult, (RUcontext* pctx, unsigned int flags, RUdevice dev));
ROCCU_DEFINE_FUNC(CtxDestroy, RU_DRIVER, cuCtxDestroy_v2, hipCtxDestroy, RUresult, (RUcontext ctx));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetCurrent, RU_DRIVER, RUresult, (RUcontext* pctx));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetDevice, RU_DRIVER, RUresult, (RUdevice* pdev));
ROCCU_DEFINE_FUNC(CtxGetStreamPriorityRange, RU_DRIVER, cuCtxGetStreamPriorityRange, hipDeviceGetStreamPriorityRange, RUresult, (int* leastPriority, int* greatestPriority));
ROCCU_DEFINE_DIRECT_FUNC(CtxSetCurrent, RU_DRIVER, RUresult, (RUcontext ctx));
ROCCU_DEFINE_FUNC(CtxSetLimit, RU_DRIVER, cuCtxSetLimit, NOOP, RUresult,(RUlimit limit, size_t value));

ROCCU_DEFINE_DIRECT_FUNC(DeviceGet, RU_DRIVER, RUresult, (RUdevice* device, int ordinal));

enum RUdevice_attribute {
    RU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    RU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    RU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    RU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    RU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    RU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    RU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    RU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    RU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
    RU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    RU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    RU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    RU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    RU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
    RU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    RU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    RU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    RU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    RU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    RU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    RU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    RU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
    RU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    RU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    RU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    RU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    RU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    RU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    RU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    RU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    RU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    RU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    RU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    RU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    RU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    RU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    RU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    RU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    RU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    RU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    RU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    RU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    RU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    RU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    RU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    RU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    RU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    RU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    RU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    RU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    RU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    RU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    RU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    RU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    RU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    RU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    RU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    RU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    RU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    RU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    RU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    RU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    RU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    RU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    RU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    RU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    RU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    RU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    RU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    RU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    RU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    RU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    RU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    RU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    RU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    RU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    RU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    RU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    RU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    RU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    RU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    RU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    RU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    RU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    RU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    RU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    RU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    RU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    RU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    RU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    RU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    RU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    RU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    RU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    RU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    RU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    RU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134
};

ROCCU_DEFINE_DIRECT_FUNC(DeviceGetAttribute, RU_DRIVER, RUresult, (int* pi, RUdevice_attribute attr, RUdevice dev));
ROCCU_DEFINE_DIRECT_FUNC(DeviceGetName, RU_DRIVER, RUresult, (char* name, int len, RUdevice dev));

ROCCU_DEFINE_DIRECT_FUNC(EventCreate, RU_DRIVER, RUresult, (RUevent* phEvent, unsigned int Flags));
ROCCU_DEFINE_FUNC(EventDestroy, RU_DRIVER, cuEventDestroy_v2, hipEventDestroy, RUresult, (RUevent phEvent));
ROCCU_DEFINE_DIRECT_FUNC(EventElapsedTime, RU_DRIVER, RUresult, (float* pMilliseconds, RUevent hStart, RUevent hEnd));
ROCCU_DEFINE_DIRECT_FUNC(EventQuery, RU_DRIVER, RUresult, (RUevent hEvent));
ROCCU_DEFINE_DIRECT_FUNC(EventRecord, RU_DRIVER, RUresult, (RUevent hEvent, RUstream hStream));
ROCCU_DEFINE_DIRECT_FUNC(EventSynchronize, RU_DRIVER, RUresult, (RUevent hEvent));

ROCCU_DEFINE_DIRECT_FUNC(FuncGetAttribute, RU_DRIVER, RUresult,(int* pi, RUfunction_attribute attrib, RUfunction hfunc));

ROCCU_DEFINE_DIRECT_FUNC(GetErrorString, RU_DRIVER, RUresult, (RUresult error, const char** pStr));
ROCCU_DEFINE_DIRECT_FUNC(GetErrorName, RU_DRIVER, RUresult, (RUresult error, const char** pStr));

ROCCU_DEFINE_DIRECT_FUNC(GraphicsGLRegisterImage, RU_DRIVER, RUresult, (RUgraphicsResource* ruResource, unsigned int image, unsigned int target, unsigned int flags));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsMapResources, RU_DRIVER, RUresult, (unsigned int count, RUgraphicsResource* resources, RUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsSubResourceGetMappedArray, RU_DRIVER, RUresult, (RUarray* pArray, RUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsUnmapResources, RU_DRIVER, RUresult, (unsigned int count, RUgraphicsResource* resources, RUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsUnregisterResource, RU_DRIVER, RUresult, (RUgraphicsResource resources));

ROCCU_DEFINE_DIRECT_FUNC(Init, RU_DRIVER, RUresult, (unsigned int Flags));

ROCCU_DEFINE_FUNC(LaunchCooperativeKernel, RU_DRIVER, cuLaunchCooperativeKernel, hipModuleLaunchCooperativeKernel, RUresult, (RUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, RUstream stream, void** kernelParams));
ROCCU_DEFINE_DIRECT_FUNC(LaunchHostFunc, RU_DRIVER, RUresult, (RUstream stream, RUhostFn func, void* userData));
ROCCU_DEFINE_FUNC(LaunchKernel, RU_DRIVER, cuLaunchKernel, hipModuleLaunchKernel, RUresult, (RUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, RUstream stream, void** kernelParams, void** extra));

ROCCU_DEFINE_FUNC(LinkAddData, RU_DRIVER, cuLinkAddData_v2, hiprtcLinkAddData, RUresult, (RUlinkState state, RUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, const char** options, void** optionValues));
ROCCU_DEFINE_FUNC(LinkComplete, RU_DRIVER, cuLinkComplete, hiprtcLinkCreate, RUresult, (RUlinkState state, void** cubinOut, size_t* sizeOut));
ROCCU_DEFINE_FUNC(LinkCreate, RU_DRIVER, cuLinkCreate_v2, hiprtcLinkCreate, RUresult, (unsigned int numOptions, void** options, void** optionValues, RUlinkState* stateOut));
ROCCU_DEFINE_FUNC(LinkDestroy, RU_DRIVER, cuLinkDestroy, hiprtcLinkDestroy, RUresult, (RUlinkState state));

ROCCU_DEFINE_FUNC(MemAlloc, RU_DRIVER, cuMemAlloc_v2, hipMalloc, RUresult, (RUdeviceptr* dptr, size_t size));
ROCCU_DEFINE_FUNC(MemAllocAsync, RU_DRIVER, cuMemAllocAsync, hipMallocAsync, RUresult, (RUdeviceptr* dptr, size_t size, RUstream stream));
ROCCU_DEFINE_FUNC(MemAllocHost, RU_DRIVER, cuMemAllocHost_v2, hipHostMalloc, RUresult, (void** pp, size_t bytes));

ROCCU_DEFINE_FUNC(Memcpy2D, RU_DRIVER, cuMemcpy2D_v2, hipMemcpy2D, RUresult, (const RU_MEMCPY2D* pCopy));
ROCCU_DEFINE_FUNC(Memcpy2DAsync, RU_DRIVER, cuMemcpy2DAsync_v2, hipMemcpy2DAsync, RUresult, (const RU_MEMCPY2D* pCopy, RUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyDtoD, RU_DRIVER, cuMemcpyDtoD_v2, hipMemcpyDtoD, RUresult, (RUdeviceptr dstDevice, RUdeviceptr srcDevice, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyDtoDAsync, RU_DRIVER, cuMemcpyDtoDAsync_v2, hipMemcpyDtoDAsync, RUresult, (RUdeviceptr dstDevice, RUdeviceptr srcDevice, size_t ByteCount, RUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyDtoH, RU_DRIVER, cuMemcpyDtoH_v2, hipMemcpyDtoH, RUresult, (void* dstHost, RUdeviceptr srcDevice, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyDtoHAsync, RU_DRIVER, cuMemcpyDtoHAsync_v2, hipMemcpyDtoHAsync, RUresult, (void* dstHost, RUdeviceptr srcDevice, size_t ByteCount, RUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyHtoD, RU_DRIVER, cuMemcpyHtoD_v2, hipMemcpyHtoD, RUresult, (RUdeviceptr dstDevice, const void* srcHost, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyHtoDAsync, RU_DRIVER, cuMemcpyHtoDAsync_v2, hipMemcpyHtoDAsync, RUresult, (RUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, RUstream hStream));

ROCCU_DEFINE_FUNC(MemFree, RU_DRIVER, cuMemFree_v2, hipFree, RUresult, (RUdeviceptr dptr));
ROCCU_DEFINE_FUNC(MemFreeAsync, RU_DRIVER, cuMemFreeAsync, hipFreeAsync, RUresult, (RUdeviceptr dptr, RUstream stream));
ROCCU_DEFINE_FUNC(MemFreeHost, RU_DRIVER, cuMemFreeHost, hipHostFree, RUresult, (void* p));

ROCCU_DEFINE_FUNC(MemsetD8, RU_DRIVER, cuMemsetD8_v2, hipMemsetD8, RUresult, (RUdeviceptr dstDevice, unsigned char uc, size_t N));
ROCCU_DEFINE_DIRECT_FUNC(MemsetD8Async, RU_DRIVER, RUresult, (RUdeviceptr dstDevice, unsigned char uc, size_t N, RUstream stream));

ROCCU_DEFINE_DIRECT_FUNC(ModuleGetFunction, RU_DRIVER, RUresult, (RUfunction* hfunc, RUmodule hmod, const char* name));
ROCCU_DEFINE_FUNC(ModuleGetGlobal, RU_DRIVER, cuModuleGetGlobal_v2, hipModuleGetGlobal, RUresult, (RUdeviceptr* dptr, size_t* bytes, RUmodule hmod, const char* name));
ROCCU_DEFINE_FUNC(ModuleLoadDataEx, RU_DRIVER, cuModuleLoadDataEx, hipModuleLoadData, RUresult, (RUmodule* module, const void* image, unsigned int numOptions, const char** options, void** optionValues));
ROCCU_DEFINE_DIRECT_FUNC(ModuleUnload, RU_DRIVER, RUresult, (RUmodule hmod));

ROCCU_DEFINE_FUNC(OccupancyMaxActiveBlocksPerMultiprocessor, RU_DRIVER, cuOccupancyMaxActiveBlocksPerMultiprocessor, hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, RUresult, (int* numBlocks, RUfunction func, int blockSize, size_t dynamicSMemSize));
ROCCU_DEFINE_FUNC(OccupancyMaxPotentialBlockSize, RU_DRIVER, cuOccupancyMaxPotentialBlockSize, hipModuleOccupancyMaxPotentialBlockSize, RUresult, (int* minGridSize, int* blockSize, RUfunction func, void* blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit));

ROCCU_DEFINE_DIRECT_FUNC(StreamCreateWithPriority, RU_DRIVER, RUresult, (RUstream* pStream, unsigned int flags, int priority));
ROCCU_DEFINE_FUNC(StreamDestroy, RU_DRIVER, cuStreamDestroy_v2, hipStreamDestroy, RUresult, (RUstream stream));
ROCCU_DEFINE_FUNC(StreamSetAttribute, RU_DRIVER, cuStreamSetAttribute, NOOP, RUresult, (RUstream stream, RUlaunchAttributeID attr, RUlaunchAttributeValue* value));
ROCCU_DEFINE_DIRECT_FUNC(StreamSynchronize, RU_DRIVER, RUresult, (RUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(StreamWaitEvent, RU_DRIVER, RUresult, (RUstream stream, RUevent event, unsigned int flags));

ROCCU_DEFINE_FUNC(rtcAddNameExpression, RU_RTC, nvrtcAddNameExpression, hiprtcAddNameExpression, rurtcResult, (rurtcProgram prog, const char* name_expression));
ROCCU_DEFINE_FUNC(rtcCompileProgram, RU_RTC, nvrtcCompileProgram, hiprtcCompileProgram, rurtcResult, (rurtcProgram prog, int numOptions, const char** options));
ROCCU_DEFINE_FUNC(rtcCreateProgram, RU_RTC, nvrtcCreateProgram, hiprtcCreateProgram, rurtcResult, (rurtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames));
ROCCU_DEFINE_FUNC(rtcDestroyProgram, RU_RTC, nvrtcDestroyProgram, hiprtcDestroyProgram, rurtcResult, (rurtcProgram* prog));
ROCCU_DEFINE_FUNC(rtcGetErrorString, RU_RTC, nvrtcGetErrorString, hiprtcGetErrorString, const char*, (rurtcResult));
ROCCU_DEFINE_FUNC(rtcGetLoweredName, RU_RTC, nvrtcGetLoweredName, hiprtcGetLoweredName, rurtcResult, (rurtcProgram prog, const char* name_expression, const char** lowered_name));
ROCCU_DEFINE_FUNC(rtcGetProgramLog, RU_RTC, nvrtcGetProgramLog, hiprtcGetProgramLog, rurtcResult, (rurtcProgram prog, char* log));
ROCCU_DEFINE_FUNC(rtcGetProgramLogSize, RU_RTC, nvrtcGetProgramLogSize, hiprtcGetProgramLogSize, rurtcResult, (rurtcProgram prog, size_t* logSizeRet));
ROCCU_DEFINE_FUNC(rtcGetAssembly, RU_RTC, nvrtcGetCUBIN, hiprtcGetCode, rurtcResult, (rurtcProgram prog, char* ptx));
ROCCU_DEFINE_FUNC(rtcGetAssemblySize, RU_RTC, nvrtcGetCUBINSize, hiprtcGetCodeSize, rurtcResult, (rurtcProgram prog, size_t* ptxSizeRet));

#undef ROCCU_DEFINE_FUNC
#undef ROCCU_DEFINE_DIRECT_FUNC
#undef ROCCU_DEFINE_OPAQUE