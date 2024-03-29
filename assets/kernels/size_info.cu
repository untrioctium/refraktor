#include <refrakt/flamelib.h>
#include <refrakt/random.h>

struct void_t {};

template<typename Real, unsigned long long ThreadsPerBlock>
using sample_state_t = flamelib::sample_state_tmpl<void_t, Real, xoroshiro64<Real>, ThreadsPerBlock>;

template<unsigned long long ThreadsPerBlock, typename Real>
__device__ unsigned long long calc_size() {
    printf("sizeof(sample_state_t<Real, %llu>) = %llu\n", ThreadsPerBlock, sizeof(sample_state_t<Real, ThreadsPerBlock>));
    return sizeof(sample_state_t<Real, ThreadsPerBlock>);
}

__global__ void get_sizes(unsigned long long* sizes) {
    sizes[0] = sizeof(flamelib::iteration_info_t);