#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void block_gemm_kernel(const float* __restrict__ A, 
                                  const float* __restrict__ B, 
                                  float* __restrict__ C, 
                                  int n) {
                                            
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    int num_tiles = n / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        s_A[ty][tx] = A[row * n + (t * BLOCK_SIZE + tx)];
        s_B[ty][tx] = B[(t * BLOCK_SIZE + ty) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    C[row * n + col] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 grid_size(n / BLOCK_SIZE, n / BLOCK_SIZE);

    block_gemm_kernel <<< grid_size, block_size >>> (d_A, d_B, d_C, n);
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}