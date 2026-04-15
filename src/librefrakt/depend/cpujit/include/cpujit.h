#ifndef CPUJIT_H
#define CPUJIT_H

#include <stddef.h>
#include <librefrakt/flame_types.hpp>
#include <librefrakt/flame_info.hpp>

#ifdef _WIN32
    #ifdef CPUJIT_BUILDING
        #define CPUJIT_API __declspec(dllexport)
    #else
        #define CPUJIT_API __declspec(dllimport)
    #endif
#else
    #define CPUJIT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cpujit_module* cpujit_module_t;

/* Compile LLVM IR text into a JIT module.
 * On success, returns a valid module handle and sets *error_out to NULL.
 * On failure, returns NULL and sets *error_out to a descriptive string.
 * The error string is valid until the next call to cpujit_compile or cpujit_error_free. */
CPUJIT_API cpujit_module_t cpujit_compile(const char* llvm_ir, const char** error_out);

/* Look up a symbol by name in a compiled module.
 * Returns a callable function pointer, or NULL if the symbol is not found. */
CPUJIT_API void* cpujit_lookup(cpujit_module_t mod, const char* symbol_name);

/* Returns LLVM IR declarations for all available math functions.
 * Prepend this to your IR before any calls to sinf, cosf, expf, etc.
 * The returned pointer is static and valid for the lifetime of the process. */
CPUJIT_API const char* cpujit_preamble(void);

/* Returns the generated native assembly text for the module.
 * The returned pointer is owned by the module and valid until cpujit_destroy. */
CPUJIT_API const char* cpujit_asm(cpujit_module_t mod);

/* Destroy a JIT module and free all associated resources.
 * Any function pointers obtained from this module become invalid. */
CPUJIT_API void cpujit_destroy(cpujit_module_t mod);

/* Free an error string returned by cpujit_compile. */
CPUJIT_API void cpujit_error_free(const char* error);

CPUJIT_API cpujit_module_t cpujit_make_flame_dispatch(const rfkt::flame* flame, const rfkt::flamedb* fdb, rfkt::precision prec, const char** error_out);

#ifdef __cplusplus
}
#endif

#endif /* CPUJIT_H */
