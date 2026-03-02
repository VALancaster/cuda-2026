#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> res(input.size());
    int n = input.size();
    int i = 0;
    float sqrt_2_div_pi = 0.7978845608028653558798921198687637369517172623298693153318516593f;

#pragma omp parallel for simd
    for (i = 0; i < n; i++) {
        res[i] = input[i] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i] + 0.044715f * input[i] * input[i] * input[i])) + 1.0f));
    }

    return res;
}
