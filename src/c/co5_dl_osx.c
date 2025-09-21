// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>

const unsigned char FLOAT_REGISTER_COUNT = 8;
const unsigned char DOUBLE_REGISTER_COUNT = 4;

unsigned char matrix_multiply_double(unsigned long n, unsigned long d, unsigned long m, double *rearranged, double *dXm, double *out)
{
    __m256d operand1 = _mm256_setzero_pd();
    __m256d operand2 = _mm256_setzero_pd();
    __m256d operand3 = _mm256_setzero_pd();
    double *tmp0 = dXm;
    double *tmp1 = rearranged;
    double *tmp2 = out;
    unsigned char displacement;
    unsigned long trigger = n * d;
    unsigned long n_counter = 0;
    unsigned long m_counter = 0;

    for (unsigned int c = 0; c < d * d; c++)
    {
        operand1 = _mm256_loadu_pd(tmp0);
        operand2 = _mm256_loadu_pd(tmp1);
        operand3 = _mm256_loadu_pd(tmp2);
        operand3 = _mm256_fmadd_pd(operand1, operand2, operand3);
        _mm256_storeu_pd(tmp2, operand3);
        m_counter += DOUBLE_REGISTER_COUNT;
        displacement = DOUBLE_REGISTER_COUNT;
        if (!(m_counter < m))
        {

            displacement = m;
            if (n_counter++ < n)
            {
                memset(tmp2 + displacement, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - m));
            }
        }
        tmp1 += displacement;
        tmp2 += displacement;
        if (n_counter == n)
        {
            tmp0 += m;
            tmp2 = out;
        }
    }
    return 0;
}

unsigned char matrix_multiply_float(unsigned long n, unsigned long d, unsigned long m, float *rearranged, float *dXm, float *out)
{
    __m256 operand1 = _mm256_setzero_ps();
    __m256 operand2 = _mm256_setzero_ps();
    __m256 operand3 = _mm256_setzero_ps();
    float *tmp0 = dXm;
    float *tmp1 = rearranged;
    float *tmp2 = out;
    unsigned char displacement;
    unsigned long n_counter = 0;
    unsigned long m_counter = 0;

    for (unsigned int c = 0; c < d * m; c++)
    {
        operand1 = _mm256_loadu_ps(tmp0);
        operand2 = _mm256_loadu_ps(tmp1);
        operand3 = _mm256_loadu_ps(tmp2);
        operand3 = _mm256_fmadd_ps(operand1, operand2, operand3);
        _mm256_storeu_ps(tmp2, operand3);
        m_counter += FLOAT_REGISTER_COUNT;
        displacement = FLOAT_REGISTER_COUNT;
        if (!(m_counter < m))
        {
            displacement = m;
            if (n_counter++ < n)
            {
                memset(tmp2 + displacement, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - m));
            }
    }
        tmp1 += displacement;
        tmp2 += displacement;
        if (n_counter == n)
        {
            tmp0 += m;
            tmp2 = out;
        }
    }
    return 0;
}
