
//Compile flags: -O3 -fopenmp -ffast-math -ftree-vectorize -funroll-loops
#pragma GCC optimize("O3,fast-math,tree-vectorize,unroll-loops")
#include "naive_gemm_omp.h"
#pragma GCC target("avx2,fma")
std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> result(n*n,0.0f);
    #pragma omp parallel for schedule(static, 32)
    for(int i=0; i<n; i++){
         for(int k=0; k<n; k++){
            for(int j=0; j<n; j++){
                result[i*n+j]+=a[i*n+k]*b[k*n+j];
            }
        }
    }
    return result;
}