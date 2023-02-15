template<typename FloatT>
struct vec2 {
	FloatT x, y;
};

#include <refrakt/flamelib.h>
#include <refrakt/random.h>

struct void_t {};

template<typename Real, unsigned long long ThreadsPerBlock>
using shared_state_t = flamelib::shared_state_tmpl<void_t, Real, xoroshiro64<Real>, ThreadsPerBlock>;

template<unsigned long long... ThreadsPerBlock>
__device__ unsigned long long required_shared[sizeof...(ThreadsPerBlock) * 2];

template<unsigned long long... ThreadsPerBlock>
__global__ void get_sizes() {

    auto& result = required_shared<ThreadsPerBlock...>;

    int idx = 0;
    ((result[idx] = sizeof(shared_state_t<float, ThreadsPerBlock>), ++idx), ...);
    ((result[idx] = sizeof(shared_state_t<double, ThreadsPerBlock>), ++idx), ...);
} 