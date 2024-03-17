#include <cstdint>

using RUdevice = int;
using RUresult = int;
using rurtcResult = int;
using RUdeviceptr = unsigned long long;
using RUhostFn = void (*)(void* userData);

#define ROCCU_DEFINE_OPAQUE(name) \
    using name = struct name##_st*

ROCCU_DEFINE_OPAQUE(RUctx);
ROCCU_DEFINE_OPAQUE(RUstream);
ROCCU_DEFINE_OPAQUE(RUmodule);
ROCCU_DEFINE_OPAQUE(RUfunction);
ROCCU_DEFINE_OPAQUE(RUgraphicsResource);
ROCCU_DEFINE_OPAQUE(RUarray);
ROCCU_DEFINE_OPAQUE(RUlinkState);
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

struct RU_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;
    size_t srcZ;
    size_t srcLOD;
    RUmemorytype srcMemoryType;
    const void* srcHost;
    RUdeviceptr srcDevice;
    RUarray srcArray;
    void* reserved0;
    size_t srcPitch;
    size_t srcHeight;

    size_t dstXInBytes;
    size_t dstY;
    size_t dstZ;
    size_t dstLOD;
    RUmemorytype dstMemoryType;
    void* dstHost;
    RUdeviceptr dstDevice;
    RUarray dstArray;
    void* reserved1;
    size_t dstPitch;
    size_t dstHeight;

    size_t WidthInBytes;
    size_t Height;
    size_t Depth;
};

enum roccu_api {
    ROCCU_API_CUDA,
    ROCCU_API_ROCM,
    ROCCU_API_NONE
};

#ifndef ROCCU_IMPL
#define ROCCU_API
#else

#include <unordered_map>

enum source_t {
    RU_DRIVER,
    RU_RTC
};

struct ru_traits {

    void** func;
    const char* name;
    source_t source;
    const char* cuda_name;
    const char* rocm_name;
};

inline static std::unordered_map<const char*, ru_traits> ru_map = {};

bool register_traits(const ru_traits& traits) {
    ru_map[traits.name] = traits;
    return true;
};

#endif

roccu_api roccuInit();
roccu_api roccuGetApi();

#ifdef ROCCU_API
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) \
        using PFN##name##PROC = RET (*)ARGS; \
        extern PFN##name##PROC ru ## name
#else 
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) \
        using PFN##name##PROC = RET (*)ARGS; \
        PFN##name##PROC ru ## name = nullptr; \
		static bool name##_init = register_traits({(void**)& ru ## name, #name, SRC, #CUDA_NAME, #ROCM_NAME});
#endif

#define ROCCU_DEFINE_DIRECT_FUNC(name, SRC, RET, ARGS) ROCCU_DEFINE_FUNC(name, SRC, cu##name, hip##name, RET, ARGS)

ROCCU_DEFINE_FUNC(CtxCreate, RU_DRIVER, cuCtxCreate_v2, hipCtxCreate, RUresult, (RUctx* pctx, unsigned int flags, RUdevice dev));
ROCCU_DEFINE_FUNC(CtxDestroy, RU_DRIVER, cuCtxDestroy_v2, hipCtxDestroy, RUresult, (RUctx ctx));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetCurrent, RU_DRIVER, RUresult, (RUctx* pctx));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetDevice, RU_DRIVER, RUresult, (RUdevice* pdev));
ROCCU_DEFINE_DIRECT_FUNC(CtxGetStreamPriorityRange, RU_DRIVER, RUresult, (int* leastPriority, int* greatestPriority));
ROCCU_DEFINE_DIRECT_FUNC(CtxSetCurrent, RU_DRIVER, RUresult, (RUctx ctx));
//ROCCU_DEFINE_DIRECT_FUNC(CtxSetLimit, RU_DRIVER, RUresult,(int limit, size_t value));

ROCCU_DEFINE_DIRECT_FUNC(DeviceGet, RU_DRIVER, RUresult, (RUdevice* device, int ordinal));
ROCCU_DEFINE_DIRECT_FUNC(DeviceGetAttribute, RU_DRIVER, RUresult, (int* pi, int attr, RUdevice dev));
ROCCU_DEFINE_DIRECT_FUNC(DeviceGetName, RU_DRIVER, RUresult, (char* name, int len, RUdevice dev));

//ROCCU_DEFINE_DIRECT_FUNC(FuncGetAttribute, RU_DRIVER, RUresult,(int* pi, RUfunction hfunc, int attrib));

ROCCU_DEFINE_DIRECT_FUNC(GetErrorString, RU_DRIVER, RUresult, (RUresult error, const char** pStr));
ROCCU_DEFINE_DIRECT_FUNC(GetErrorName, RU_DRIVER, RUresult, (RUresult error, const char** pStr));

ROCCU_DEFINE_DIRECT_FUNC(GraphicsGLRegisterImage, RU_DRIVER, RUresult, (RUgraphicsResource* ruResource, unsigned int image, unsigned int target, unsigned int flags));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsMapResources, RU_DRIVER, RUresult, (unsigned int count, RUgraphicsResource* resources, RUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsSubResourceGetMappedArray, RU_DRIVER, RUresult, (RUarray* pArray, RUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsUnmapResources, RU_DRIVER, RUresult, (unsigned int count, RUgraphicsResource* resources, RUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(GraphicsUnregisterResource, RU_DRIVER, RUresult, (RUgraphicsResource resources));

ROCCU_DEFINE_DIRECT_FUNC(Init, RU_DRIVER, RUresult, (unsigned int Flags));

ROCCU_DEFINE_DIRECT_FUNC(LaunchCooperativeKernel, RU_DRIVER, RUresult, (RUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, RUstream stream, void** kernelParams));
ROCCU_DEFINE_DIRECT_FUNC(LaunchHostFunc, RU_DRIVER, RUresult, (RUstream stream, RUhostFn func, void* userData));
ROCCU_DEFINE_DIRECT_FUNC(LaunchKernel, RU_DRIVER, RUresult, (RUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, RUstream stream, void** kernelParams, void** extra));

ROCCU_DEFINE_FUNC(LinkAddData, RU_DRIVER, cuLinkAddData_v2, hipLinkAddData, RUresult, (RUlinkState state, RUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, const char** options, void** optionValues));
ROCCU_DEFINE_DIRECT_FUNC(LinkComplete, RU_DRIVER, RUresult, (RUlinkState state, void** cubinOut, size_t* sizeOut));
ROCCU_DEFINE_FUNC(cuLinkCreate, RU_DRIVER, cuLinkCreate_v2, hipLinkCreate, RUresult, (unsigned int numOptions, void** options, void** optionValues, RUlinkState* stateOut));

ROCCU_DEFINE_FUNC(MemAlloc, RU_DRIVER, cuMemAlloc_v2, hipMemAlloc, RUresult, (RUdeviceptr* dptr, size_t size));
ROCCU_DEFINE_DIRECT_FUNC(MemAllocAsync, RU_DRIVER, RUresult, (RUdeviceptr* dptr, size_t size, RUstream stream));
ROCCU_DEFINE_FUNC(MemAllocHost, RU_DRIVER, cuMemAllocHost_v2, hipHostMalloc, RUresult, (void** pp, size_t bytes));

ROCCU_DEFINE_FUNC(Memcpy2D, RU_DRIVER, cuMemcpy2D_v2, hipMemcpy2D, RUresult, (const RU_MEMCPY2D* pCopy));
ROCCU_DEFINE_FUNC(Memcpy2DAsync, RU_DRIVER, cuMemcpy2DAsync_v2, hipMemcpy2DAsync, RUresult, (const RU_MEMCPY2D* pCopy, RUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyDtoD, RU_DRIVER, cuMemcpyDtoD_v2, hipMemcpyDtoD, RUresult, (RUdeviceptr dstDevice, RUdeviceptr srcDevice, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyDtoH, RU_DRIVER, cuMemcpyDtoH_v2, hipMemcpyDtoH, RUresult, (void* dstHost, RUdeviceptr srcDevice, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyDtoHAsync, RU_DRIVER, cuMemcpyDtoHAsync_v2, hipMemcpyDtoHAsync, RUresult, (void* dstHost, RUdeviceptr srcDevice, size_t ByteCount, RUstream hStream));
ROCCU_DEFINE_FUNC(MemcpyHtoD, RU_DRIVER, cuMemcpyHtoD_v2, hipMemcpyHtoD, RUresult, (RUdeviceptr dstDevice, const void* srcHost, size_t ByteCount));
ROCCU_DEFINE_FUNC(MemcpyHtoDAsync, RU_DRIVER, cuMemcpyHtoDAsync_v2, hipMemcpyHtoDAsync, RUresult, (RUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, RUstream hStream));

ROCCU_DEFINE_FUNC(MemFree, RU_DRIVER, cuMemFree_v2, hipMemFree, RUresult, (RUdeviceptr dptr));
ROCCU_DEFINE_DIRECT_FUNC(MemFreeAsync, RU_DRIVER, RUresult, (RUdeviceptr dptr, RUstream stream));
ROCCU_DEFINE_FUNC(MemFreeHost, RU_DRIVER, cuMemFreeHost, hipHostFree, RUresult, (void* p));

ROCCU_DEFINE_FUNC(MemsetD8, RU_DRIVER, cuMemsetD8_v2, hipMemsetD8, RUresult, (RUdeviceptr dstDevice, unsigned char uc, size_t N));
ROCCU_DEFINE_DIRECT_FUNC(MemsetD8Async, RU_DRIVER, RUresult, (RUdeviceptr dstDevice, unsigned char uc, size_t N, RUstream stream));

ROCCU_DEFINE_DIRECT_FUNC(ModuleGetFunction, RU_DRIVER, RUresult, (RUfunction* hfunc, RUmodule hmod, const char* name));
ROCCU_DEFINE_FUNC(ModuleGetGlobal, RU_DRIVER, cuModuleGetGlobal_v2, hipModuleGetGlobal, RUresult, (RUdeviceptr* dptr, size_t* bytes, RUmodule hmod, const char* name));
ROCCU_DEFINE_FUNC(ModuleLoadDataEx, RU_DRIVER, cuModuleLoadDataEx, hipModuleLoadData, RUresult, (RUmodule* module, const void* image, unsigned int numOptions, const char** options, void** optionValues));
ROCCU_DEFINE_DIRECT_FUNC(ModuleUnload, RU_DRIVER, RUresult, (RUmodule hmod));

ROCCU_DEFINE_DIRECT_FUNC(OccupancyMaxActiveBlocksPerMultiprocessor, RU_DRIVER, RUresult, (int* numBlocks, RUfunction func, int blockSize, size_t dynamicSMemSize));
ROCCU_DEFINE_DIRECT_FUNC(OccupancyMaxPotentialBlockSize, RU_DRIVER, RUresult, (int* minGridSize, int* blockSize, RUfunction func, size_t dynamicSMemSize, int blockSizeLimit));

ROCCU_DEFINE_DIRECT_FUNC(StreamCreateWithPriority, RU_DRIVER, RUresult, (RUstream* pStream, unsigned int flags, int priority));
ROCCU_DEFINE_FUNC(StreamDestroy, RU_DRIVER, cuStreamDestroy_v2, hipStreamDestroy, RUresult, (RUstream stream));
ROCCU_DEFINE_DIRECT_FUNC(StreamSynchronize, RU_DRIVER, RUresult, (RUstream stream));

ROCCU_DEFINE_FUNC(rtcAddNameExpression, RU_RTC, nvrtcAddNameExpression, hiprtcAddNameExpression, rurtcResult, (rurtcProgram prog, const char* name_expression));
ROCCU_DEFINE_FUNC(rtcCompileProgram, RU_RTC, nvrtcCompileProgram, hiprtcCompileProgram, rurtcResult, (rurtcProgram prog, int numOptions, const char** options));
ROCCU_DEFINE_FUNC(rtcCreateProgram, RU_RTC, nvrtcCreateProgram, hiprtcCreateProgram, rurtcResult, (rurtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames));
ROCCU_DEFINE_FUNC(rtcDestroyProgram, RU_RTC, nvrtcDestroyProgram, hiprtcDestroyProgram, rurtcResult, (rurtcProgram* prog));
ROCCU_DEFINE_FUNC(rtcGetErrorString, RU_RTC, nvrtcGetErrorString, hiprtcGetErrorString, const char*, (rurtcResult));
ROCCU_DEFINE_FUNC(rtcGetLoweredName, RU_RTC, nvrtcGetLoweredName, hiprtcGetLoweredName, rurtcResult, (rurtcProgram prog, const char* name_expression, const char** lowered_name));
ROCCU_DEFINE_FUNC(rtcGetProgramLog, RU_RTC, nvrtcGetProgramLog, hiprtcGetProgramLog, rurtcResult, (rurtcProgram prog, char* log));
ROCCU_DEFINE_FUNC(rtcGetProgramLogSize, RU_RTC, nvrtcGetProgramLogSize, hiprtcGetProgramLogSize, rurtcResult, (rurtcProgram prog, size_t* logSizeRet));
ROCCU_DEFINE_FUNC(rtcGetAssembly, RU_RTC, nvrtcGetPTX, hiprtcGetPTX, rurtcResult, (rurtcProgram prog, const char** ptx));
ROCCU_DEFINE_FUNC(rtcGetAssemblySize, RU_RTC, nvrtcGetPTXSize, hiprtcGetPTXSize, rurtcResult, (rurtcProgram prog, size_t* ptxSizeRet));