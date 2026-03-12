#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

// ядро для вычисления GELU активации на каждом ядре GPU
__global__ void GeluKernel(const float4* input, float4* output, int n_vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс потока

    if (i < n_vec) {
        float4 x4 = input[i]; // считывание сразу 4 чисел
        float4 res;

        const float double_sqrt_2_over_pi = 1.59576912f; // 2 * sqrt(2 / pi)
        const float coeff = 0.044715f;

        // Вычисление GELU для каждого элемента вектора
        float argX = double_sqrt_2_over_pi * (x4.x + coeff * x4.x * x4.x * x4.x);
        res.x = x4.x / (1.0f + expf(-argX)); // формула, аппроксимирующая GELU
        float argY = double_sqrt_2_over_pi * (x4.y + coeff * x4.y * x4.y * x4.y);
        res.y = x4.y / (1.0f + expf(-argY)); // формула, аппроксимирующая GELU
        float argZ = double_sqrt_2_over_pi * (x4.z + coeff * x4.z * x4.z * x4.z);
        res.z = x4.z / (1.0f + expf(-argZ)); // формула, аппроксимирующая GELU
        float argW = double_sqrt_2_over_pi * (x4.w + coeff * x4.w * x4.w * x4.w);
        res.w = x4.w / (1.0f + expf(-argW)); // формула, аппроксимирующая GELU

        output[i] = res; // запись сразу 4 чисел обратно в память
    }
}

// основная функция (выполняется на CPU)
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    int n = static_cast<int>(input.size());

    // статические указатели, чтобы не делать аллокацию и деаллоцацию памяти на каждом вызове
    static float *d_in = nullptr;
    static float *d_out = nullptr;
    static int allocated_size = 0;
    static cudaStream_t stream = nullptr;

    // выделение памяти на GPU
    if (allocated_size < n) {
        if (d_in)
            cudaFree(d_in);
        if (d_out)
            cudaFree(d_out);
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        if (!stream)
            cudaStreamCreate(&stream); // создание потока для асинхронных операций
        allocated_size = n;
    }

    cudaMemcpyAsync(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream); // копируем данные с CPU на GPU асинхронно в потоке stream

    // настройка сетки для float4
    int n_vec = n / 4;
    int threadsPerBlock = 256; 
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; 

    GeluKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>((float4*)d_in, (float4*)d_out, n_vec); // запуск ядра на GPU с конфигурацией запуска асинхронно в потоке stream

    std::vector<float> output(n); // пока GPU выполняет вычисления, выделяем память для результата на CPU

    cudaMemcpyAsync(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost, stream); // копируем результат обратно на CPU асинхронно в потоке stream

    cudaStreamSynchronize(stream); // синхронизация всех операций в потоке stream 

    return output;
}