#define USE_ASYNC_MEMCPY

#include <refrakt/flamelib.h>
#include <refrakt/random.h>

struct void_t { char byte; };

template<typename Real, unsigned long long ThreadsPerBlock>
using shared_state_t = flamelib::shared_state_tmpl<void_t, Real, xoroshiro64<Real>, ThreadsPerBlock>;

template<unsigned long long Size>
__device__ unsigned long long required_shared[Size];

template<unsigned long long ThreadsPerBlock, typename Real>
unsigned long long calc_size() {
    return sizeof(shared_state_t<Real, ThreadsPerBlock>);
}

template<unsigned long long... ThreadsPerBlock>
__global__ void get_sizes() {

    auto& result = required_shared<sizeof...(ThreadsPerBlock) * 2>;

    int idx = 0;
    ((result[idx] = calc_size<ThreadsPerBlock, float>(), ++idx), ...);
    ((result[idx] = calc_size<ThreadsPerBlock, double>(), ++idx), ...);

} 