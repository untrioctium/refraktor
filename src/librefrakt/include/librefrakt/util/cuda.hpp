#include <roccu_cpp_types.hpp>

namespace rfkt {
    namespace cuda {
        auto init() -> roccu::context;
        auto init(int device_ordinal) -> roccu::context;
    }
}