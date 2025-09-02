//Copyright (c) 2025, tree-chutes

#include <immintrin.h>

double* kernel;
unsigned long output_shape;
unsigned long weights_width;

void multiply(double *in, double *out)
{
    double* tmp1;
    __m256d operand1 = _mm256_setzero_pd();
    __m256d operand2 = _mm256_setzero_pd();
    __m256d ret = _mm256_setzero_pd();
    double* tmp0 = in;
    double* tmp2 = out;

    for(unsigned long c = 0; c < output_shape * weights_width; c++ )
    {
        if (c % weights_width == 0)
        {
            tmp1 = kernel;
        }
        operand1 = _mm256_loadu_pd(tmp0);
        operand2 = _mm256_loadu_pd(tmp1);
        ret = _mm256_mul_pd(operand1, operand2);
        _mm256_storeu_pd(tmp2, ret);
        tmp0 += weights_width;
        tmp1 += weights_width;
        tmp2 += weights_width;
    }
}

unsigned char init_kernel(unsigned long o, unsigned long w, double *k)
{
    output_shape = o;
    weights_width = w;
    kernel = k;
    return 0;
}

unsigned char drop()
{
    free(kernel);
    return 0;
}
