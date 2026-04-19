#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    __shared__ float a_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_tile[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int k = 0; k < n; k += BLOCK_SIZE) {
        if (row < n && (k + threadIdx.x) < n) {
            a_tile[threadIdx.y][threadIdx.x] = a[row * n + (k + threadIdx.x)];
        } else {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < n && (k + threadIdx.y) < n) {
            b_tile[threadIdx.y][threadIdx.x] = b[(k + threadIdx.y) * n + col];
        } else {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; ++kk) {
            sum += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

static float *d_a = nullptr;
static float *d_b = nullptr;
static float *d_c = nullptr;
static size_t allocated = 0;

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = n * n * sizeof(float);
    
    if (allocated < bytes) {
        if (d_a) {
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        allocated = bytes;
    }
    
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    block_gemm_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    
    return c;
}