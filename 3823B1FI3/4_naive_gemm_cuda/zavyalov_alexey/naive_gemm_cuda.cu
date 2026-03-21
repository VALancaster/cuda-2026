#include "naive_gemm_cuda.h"
#define BLOCK_SIZE 256

#pragma GCC optimize("O3,fast-math,unroll-loops")
#pragma GCC target("fma,avx2")

__global__ void gelu_kernel(const float* a, const float* b, float* result, int i, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float local_res = 0.0f;
    if (j < n) 
        for (int k = 0; k < n; k++) {
            local_res += a[i * n + k] * b[k * n + j];
        }
    result[i * n + j] = local_res;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> res(n*n);
    // const int block_size = 128;
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    static float *a_gpu_first_half = nullptr, *b_gpu = nullptr, *result_gpu_first_half = nullptr;
    static float *a_gpu_second_half = nullptr,  *result_gpu_second_half = nullptr;

    static int call_number = 0;
    static int device_capacity = 0;
    
    if (device_capacity != n * n) {
        if (device_capacity > 0) {
            cudaFree(a_gpu_first_half);
            cudaFree(b_gpu);
            cudaFree(result_gpu_first_half);
        }
        cudaMalloc(&a_gpu_first_half, n * n * sizeof(float));
        cudaMalloc(&b_gpu, n * n * sizeof(float));
        cudaMalloc(&result_gpu_first_half, n * n * sizeof(float));
        device_capacity = n * n;
    }

    cudaMemcpy(a_gpu_first_half, a.data(), n * n / 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, a.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(b_gpu_first_half, b.data(), n * n / 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(result_gpu_first_half, 0, n * n / 2);
    cudaMemset(result_gpu_second_half, 0, n * n / 2);

    cudaStream_t strm;
    cudaStreamCreate(&strm);
    cudaMemcpyAsync(a_gpu_second_half, a.data() + n * n / 2, n * n / 2 * sizeof(float), cudaMemcpyHostToDevice, strm);
    //cudaMemcpyAsync(b_gpu_second_half, b.data() + n * n / 2, n * n / 2 * sizeof(float), cudaMemcpyHostToDevice, strm);

    for (int i = 0; i < n / 2; i++) {
        gelu_kernel<<<num_blocks, BLOCK_SIZE>>>(a_gpu_first_half, b_gpu, result_gpu_first_half, i, n);
    }

    cudaMemcpy(res.data(), result_gpu_first_half, n * n / 2 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(strm);

    for (int i = n / 2; i < n; i++) {
        gelu_kernel<<<num_blocks, BLOCK_SIZE>>>(a_gpu_second_half, b_gpu, result_gpu_second_half, i, n);
    }

    cudaMemcpy(res.data() + n * n / 2, result_gpu_second_half, n * n / 2 * sizeof(float), cudaMemcpyDeviceToHost);


    ++call_number;
    if (call_number == 5) {
        cudaFree(a_gpu_first_half);
        cudaFree(b_gpu);
        cudaFree(result_gpu_first_half);
    }
    return res;
}