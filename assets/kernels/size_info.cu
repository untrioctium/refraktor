#include <refrakt/flamelib.h>
#include <refrakt/random.h>

struct void_t { char byte; };

template<typename Real, unsigned long long ThreadsPerBlock>
using shared_state_t = flamelib::shared_state_tmpl<void_t, Real, xoroshiro64<Real>, ThreadsPerBlock>;

template<unsigned long long ThreadsPerBlock, typename Real>
__device__ consteval unsigned long long calc_size() {
    return sizeof(shared_state_t<Real, ThreadsPerBlock>);
}

__global__ void get_sizes(unsigned long long* sizes) {
