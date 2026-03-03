#include "block_gemm_omp.h"
#pragma GCC optimize("O3,fast-math,tree-vectorize,unroll-loops")
std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    const int block = 32;
    std::vector<float> result(n*n,0.0f);
    const float * a_ptr = a.data();
    const float * b_ptr = b.data();
    float * result_ptr = result.data();

    #pragma omp parallel for
    for (int ii=0; ii < n; ii+=block){
        for(int jj=0; jj < n; jj+=block){
            for(int kk=0; kk < n; kk+=block){
                int i_end = std::min(n,ii+block);
                int j_end = std::min(n,jj+block);
                int k_end = std::min(n,kk+block);

                for(int i = ii; i < i_end; i++){
                    float * result_str = result_ptr +i * n;
                    const float * a_str = a_ptr + i*n;

                    for(int k = kk; k < k_end; k++){
                        float a_tmp = a_str[k];
                        const float * b_str = b_ptr + k*n;
                        #pragma omp simd aligned(result_str, b_str: block)
                        for(int j = jj; j < j_end ; j++){
                            result_str[j]+=a_tmp*b_str[j];
                            //result[i*n+j]+=a_tmp*b[k*n+j];
                        }
                    }
                }
            }
        }
    }
    return result;
}