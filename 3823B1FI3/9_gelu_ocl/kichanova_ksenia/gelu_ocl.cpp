#define CL_TARGET_OPENCL_VERSION 300
#include "gelu_ocl.h"
#include <CL/cl.h>
#include <cstring>

const char* gelu_kernel_source = R"(
__kernel void gelu_kernel(__global const float* input,
                          __global float* output,
                          const int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float x3 = x * x * x;
    float p = 0.7978845608028654f * (x + 0.044715f * x3);
    
    float exp_2p = exp(2.0f * p);
    float tanh_approx = (exp_2p - 1.0f) / (exp_2p + 1.0f);
    
    output[idx] = 0.5f * x * (1.0f + tanh_approx);
}
)";

static cl_context context = nullptr;
static cl_command_queue queue = nullptr;
static cl_kernel kernel = nullptr;
static cl_mem d_input = nullptr;
static cl_mem d_output = nullptr;
static size_t allocated = 0;

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    cl_int err;
    
    if (!context) {
        cl_platform_id platform_id;
        clGetPlatformIDs(1, &platform_id, nullptr);
        
        cl_device_id device;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        
        cl_program program = clCreateProgramWithSource(context, 1, &gelu_kernel_source, nullptr, &err);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu_kernel", &err);
        clReleaseProgram(program);
    }
    
    if (allocated < bytes) {
        if (d_input) clReleaseMemObject(d_input);
        if (d_output) clReleaseMemObject(d_output);
        d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &err);
        d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        allocated = bytes;
    }
    
    clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &n);
    
    size_t global_size = n;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    
    std::vector<float> output(n);
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    
    return output;
}