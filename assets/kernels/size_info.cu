#include <refrakt/flamelib.h>
#include <refrakt/random.h>

struct void_t {};

template<typename Real, unsigned long long ThreadsPerBlock>
using sample_state_t = flamelib::sample_state_tmpl<void_t, Real, xoroshiro64<Real>, ThreadsPerBlock>;

#define COMMA ,
#define offsetof(st, m) ((size_t)&(((st *)0)->m))

template<unsigned long long ThreadsPerBlock, typename Real>
__device__ unsigned long long calc_size() {
    return offsetof(sample_state_t<Real COMMA ThreadsPerBlock>, flame);
}

__global__ void get_sizes(unsigned long long* sizes) {
    sizes[0] = sizeof(flamelib::iteration_info_t);