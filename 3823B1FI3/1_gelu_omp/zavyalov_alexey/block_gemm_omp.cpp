#include "block_gemm_omp.h"
#define BLOCK_WIDTH 1024
#define BLOCK_HEIGHT 2

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n);

#pragma omp parallel for
    for (int block_row = 0; block_row < n / BLOCK_HEIGHT; block_row++) {
        for (int block_column = 0; block_column < n / BLOCK_WIDTH; ++block_column) {
            int next_block_row = ((block_row + 1) * BLOCK_HEIGHT);
            
            for (int a_ind = block_row * BLOCK_HEIGHT; a_ind < next_block_row; a_ind++) {
                for (int k = 0; k < n; k++) {
                    int a_ind_mult_n = a_ind * n;
                    int k_mult_n = k * n;
                    int next_block_column = ((block_column + 1) * BLOCK_WIDTH);

                    // #pragma omp simd
                    for (int b_ind = block_column * BLOCK_WIDTH; b_ind < next_block_column; b_ind += 8) {
                        c[a_ind_mult_n + b_ind] += a[a_ind_mult_n + k] * b[k_mult_n + b_ind];
                        c[a_ind_mult_n + (b_ind + 1)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 1)];
                        c[a_ind_mult_n + (b_ind + 2)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 2)];
                        c[a_ind_mult_n + (b_ind + 3)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 3)];
                        c[a_ind_mult_n + (b_ind + 4)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 4)];
                        c[a_ind_mult_n + (b_ind + 5)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 5)];
                        c[a_ind_mult_n + (b_ind + 6)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 6)];
                        c[a_ind_mult_n + (b_ind + 7)] += a[a_ind_mult_n + k] * b[k_mult_n + (b_ind + 7)];
                    }
                }
            }
        }

    }

    return c;
}