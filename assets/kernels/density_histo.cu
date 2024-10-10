template<unsigned int Granularity>
__global__ void calculate_histogram(float4* input_array, int array_size, unsigned long long* histogram, unsigned long long* max_density) {

    constexpr static auto histo_bits = Granularity * 24 + 1;
    const float log_div = Granularity / log(2.0f);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned long long shared_histogram[histo_bits];
    __shared__ unsigned long long max_density_shared;

    if (threadIdx.x == 0) {
        max_density_shared = 0;
    }

    if (threadIdx.x < histo_bits) {
        shared_histogram[threadIdx.x] = 0;
    }

    __syncthreads();

    if (tid < array_size) {
        float value = input_array[tid].w;
        // Find the nearest power of two
        if(value >= 1) {
            unsigned long long bin = static_cast<unsigned long long>(log(value) * log_div) + 1;
            atomicAdd(&shared_histogram[bin], 1);
            atomicMax(&max_density_shared, static_cast<unsigned long long>(value));
        } else {
            atomicAdd(&shared_histogram[0], 1);
        }
    }

    __syncthreads();
    if (threadIdx.x < histo_bits) {
        atomicAdd(&histogram[threadIdx.x], shared_histogram[threadIdx.x]);
    }

    if (threadIdx.x == 0) {
        atomicMax(max_density, max_density_shared);
    }
}