#include "naive_gemm_cuda.h"
#define BLOCK_SIZE 256

#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("fma,avx2")


__global__ void gelu_kernel(const float* a, const float* b, float* result,  int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    float local_res = 0.0f;
    
    if (j < n && i < n) {
        local_res = 0.0f;
        for (int k = 0; k < n; k++) {
            local_res += a[i * n + k] * b[k * n + j];
        }
        result[i * n + j] = local_res;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> res(n*n);
    
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    static float *a_gpu = nullptr, *b_gpu = nullptr, *result_gpu=nullptr;

    static int call_number = 0;
    static int device_capacity = 0;
    
    if (device_capacity != n * n) {
        if (device_capacity > 0) {
            cudaFree(a_gpu);
            cudaFree(b_gpu);
            cudaFree(result_gpu);
        }
        cudaMalloc(&b_gpu, n * n * sizeof(float));
        cudaMalloc(&a_gpu, n * n * sizeof(float));
        cudaMalloc(&result_gpu, n * n  * sizeof(float));
        device_capacity = n * n;
    }

    cudaMemcpy(b_gpu, b.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_gpu, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(b_gpu_first_half, b.data(), n * n / 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(result_gpu, 0, n * n  * sizeof(float));

    //cudaMemcpyAsync(b_gpu_second_half, b.data() + n * n / 2, n * n / 2 * sizeof(float), cudaMemcpyHostToDevice, strm);

    dim3 grid((n + 31) / 32, (n + 7) / 8);
    dim3 block(32, 8);
    gelu_kernel<<<grid, block>>>(a_gpu, b_gpu, result_gpu, n);



    cudaMemcpy(res.data(), result_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost);



    // gelu_kernel<<<grid, BLOCK_SIZE>>>(a_gpu_second_half, b_gpu, result_gpu_second_half, n);


    //cudaMemcpy(res.data() + n * n / 2, result_gpu_second_half, n * n / 2 * sizeof(float), cudaMemcpyDeviceToHost);


    ++call_number;
    if (call_number == 5) {
        cudaFree(b_gpu);
    }
    return res;
}