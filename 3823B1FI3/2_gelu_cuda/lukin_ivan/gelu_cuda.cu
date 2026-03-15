#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#pragma GCC optimize("O3")

const int count = 134'217'728;

__constant__ float PI_coeff = 0.797884F;
__constant__ float coeff = 0.044715F;

const int block_size = 256;

//__restrict__ - указание nvcc о том, что данные доступны только через этот указатель в данной области видимости. Тем самым мы уменьшаем количество проверок с его стороны
__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, const int n)
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x; //получаем позицию текущего потока
    if(id >= n) return; //если вдруг потоков будет больше, чем входных данных
    
    float x = input[id];
    float exp_value = __expf(2.0F * PI_coeff * (x + coeff * x * x * x));
    float tanh_value = 1.0F - 2.0F / (exp_value + 1.0F);
    output[id] = 0.5F * x * (1.0F + tanh_value);
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    const int size = input.size();
    const int num_blocks = (size + block_size - 1) / block_size;

    float *input_gpu;
    float *output_gpu;
    const int bytes = size * sizeof(float);

    cudaMalloc((void**)&input_gpu, bytes);
    cudaMalloc((void**)&output_gpu, bytes);

    cudaMemcpy((void*)input_gpu, (void*)input.data(), bytes, cudaMemcpyHostToDevice);
    gelu_kernel <<<num_blocks, block_size>>> (input_gpu, output_gpu, size);

    std::vector<float> output(size);
    cudaMemcpy((void*)output.data(), (void*)output_gpu, bytes, cudaMemcpyDeviceToHost);

    cudaFree((void*)input_gpu);
    cudaFree((void*)output_gpu);

    return output;
}

int main()
{
    std::vector<float> input(count, 0.0F);
    std::vector<float> output(count);
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        output = GeluCUDA(input);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    int i = 0;
    std::cout << output[i += 10'000'000] << " ";
    std::cout << std::endl;
    double time = *std::min_element(time_list.begin(), time_list.end());
    std::cout << "Time: " << time <<std::endl;
}