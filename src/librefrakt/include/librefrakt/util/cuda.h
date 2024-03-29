#pragma once

#include <roccu_cpp_types.h>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <optional>
#include <array>

#include <librefrakt/util/filesystem.h>
#include <librefrakt/util.h>

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    RUresult result = x;                                          \
    if (result != RU_SUCCESS) {                                 \
      const char *msg;                                            \
      ruGetErrorName(result, &msg);                               \
      printf("`%s` failed with result: %s\n", #x, msg);             \
      printf("%s", rfkt::stacktrace().c_str());                   \
      __debugbreak();                                             \
      exit(1);                                                    \
    }                                                             \
  } while(0)

using float16 = std::uint16_t;

using half3 = ushort3;
using half4 = ushort4;

namespace rfkt::cuda {

    roccu::context init();

}