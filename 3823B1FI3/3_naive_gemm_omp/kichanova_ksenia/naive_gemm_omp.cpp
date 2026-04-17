#include "naive_gemm_omp.h"
#include <omp.h>

#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx2,fma")

std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);
    
    std::vector<float> b_t(n * n);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b_t[j * n + i] = b[i * n + j];
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float* c_row = &c[i * n];
        const float* a_row = &a[i * n];
        
        for (int j = 0; j < n; ++j) {
            const float* b_col = &b_t[j * n];
            float sum = 0.0f;
            
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k) {
                sum += a_row[k] * b_col[k];
            }
            c_row[j] = sum;
        }
    }
    
    return c;
}