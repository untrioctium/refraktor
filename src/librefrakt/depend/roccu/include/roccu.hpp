#pragma once

#include <roccu_cuda_types.h>

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
	else if constexpr(std::is_same_v<T, nvrtcResult>) return positive ? NVRTC_SUCCESS : NVRTC_ERROR_INTERNAL_ERROR;
    else if constexpr (std::is_same_v<T, const char*>) return positive ? "" : "Roccu not initialized";
}

template<bool Positive, typename Ret, typename... Args>
consteval auto get_noop(Ret(*)(Args...)) {
	return +[](Args...) -> Ret { 
        constexpr auto ret = noop_ret<Ret>(Positive);
        return static_cast<Ret>(ret); 
    };
}

template<typename FunctionPtrType>
bool register_traits(const ru_traits& traits) {

    ru_map[traits.name] = traits;
    ru_map[traits.name].noop = reinterpret_cast<void*>(get_noop<true>((FunctionPtrType)nullptr));
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
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) extern RET(*name)ARGS
#else 
    #define ROCCU_DEFINE_FUNC(name, SRC, CUDA_NAME, ROCM_NAME, RET, ARGS) \
        RET(*name)ARGS = get_noop<false>((RET(*)ARGS)nullptr); \
        RET(*name ## _dllsym)ARGS = nullptr; \
		static bool name##_init = register_traits<RET(*)ARGS>({(void**)&name, (void**)& name ## _dllsym, #name, SRC, #CUDA_NAME, #ROCM_NAME});
#endif

#include <roccu_function_table.h>

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)


#undef ROCCU_DEFINE_FUNC

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
        #if ROCCU_HAS_STACKTRACE
        cuda_error(CUresult result, const char* description = nullptr, const std::stacktrace& stacktrace = std::stacktrace::current()) : std::runtime_error(make_error_string(result, description, stacktrace)), result_(result), stacktrace_(stacktrace) {}
        #else
        cuda_error(CUresult result, const char* description = nullptr) : std::runtime_error(make_error_string(result, description)), result_(result) {}
        #endif

        #if ROCCU_HAS_STACKTRACE
        const std::stacktrace& stacktrace() const { return stacktrace_; }
        #endif

        CUresult code() const { return result_; }

    private:

        static std::string make_error_string(CUresult result, const char* description = nullptr, const std::stacktrace& stacktrace = std::stacktrace::current()) {
            std::string error_string;

            const char* cuda_name = nullptr;
            cuGetErrorName(result, &cuda_name);
            if(cuda_name) error_string += cuda_name;
            else error_string += "Unknown error";

            if(!description) return error_string;
            error_string += " caused by '";
            error_string += description;
            error_string += "'";

            #if ROCCU_HAS_STACKTRACE
            error_string += "\nStack trace:\n";
            for(const auto& frame : stacktrace) {
                error_string += frame.source_file() + ":" + std::to_string(frame.source_line()) + " " + frame.description() + "\n";
            }
            #endif
            
            return error_string;
        }

        CUresult result_;

        #if ROCCU_HAS_STACKTRACE
        std::stacktrace stacktrace_;
        #endif
    };

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    #define ROCCU_SAFE_CALL(x) do { if(auto result = x; result != CUDA_SUCCESS) throw roccu::cuda_error(result, #x); } while(0)

}
#endif