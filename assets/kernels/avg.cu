__global__ void get_quality(float4* __restrict__ bins, unsigned int dims_x, unsigned int dims_y, float* out) {
    const uint2 pos = {
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y
    };

if (pos.x >= dims_x || pos.y >= dims_y) return;
}