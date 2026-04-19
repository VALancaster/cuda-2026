#include "block_gemm_omp.h"
#include <omp.h>
#include <cstring>

#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> c(n * n, 0.0f);
    
    const int BLOCK_SIZE = 64;
    
    #pragma omp parallel
    {
        std::vector<float> local_block(BLOCK_SIZE * BLOCK_SIZE);
        
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int ii = 0; ii < n; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                const int i_end = std::min(ii + BLOCK_SIZE, n);
                const int j_end = std::min(jj + BLOCK_SIZE, n);
                
                std::fill(local_block.begin(), local_block.end(), 0.0f);
                
                for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
                    const int k_end = std::min(kk + BLOCK_SIZE, n);
                    
                    for (int i = ii; i < i_end; ++i) {
                        const float* a_row = &a[i * n];
                        float* local_row = &local_block[(i - ii) * BLOCK_SIZE];
                        
                        for (int k = kk; k < k_end; ++k) {
                            const float a_ik = a_row[k];
                            const float* b_row = &b[k * n];
                            
                            #pragma omp simd
                            for (int j = jj; j < j_end; ++j) {
                                local_row[j - jj] += a_ik * b_row[j];
                            }
                        }
                    }
                }
                
                for (int i = ii; i < i_end; ++i) {
                    std::memcpy(&c[i * n + jj], &local_block[(i - ii) * BLOCK_SIZE], (j_end - jj) * sizeof(float));
                }
            }
        }
    }
    
    return c;
}
