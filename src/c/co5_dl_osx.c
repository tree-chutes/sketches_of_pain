// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>

const unsigned char FLOAT_REGISTER_COUNT = 8;
const unsigned char DOUBLE_REGISTER_COUNT = 4;

unsigned char linear_double(unsigned long n, unsigned long d, unsigned long m, unsigned char add_biases, double *multiplier, double *dXm, double *b, double *out)
{
    __m256d operand1 = _mm256_setzero_pd();
    __m256d operand2 = _mm256_setzero_pd();
    __m256d operand3 = _mm256_setzero_pd();
    double *tmp0 = dXm;
    double *tmp1 = multiplier;
    double *tmp2 = out;
    double *tmp3 = b;
    unsigned char displacement;
    unsigned long trigger = n * d;
    unsigned long n_counter = 0;
    unsigned long m_counter = 0;

    for (unsigned int c = 0; c < d * m; c++)
    {
        operand1 = _mm256_loadu_pd(tmp0);
        operand2 = _mm256_loadu_pd(tmp1);
        operand3 = _mm256_loadu_pd(tmp2);
        operand3 = _mm256_fmadd_pd(operand1, operand2, operand3);
        if (add_biases)
        {
            operand1 = _mm256_loadu_ps(tmp3);
            operand3 = _mm256_add_ps(operand1, operand3);
        }
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
            else
            {
                add_biases = 0;
            }
        }
        tmp1 += displacement;
        tmp2 += displacement;
        if (add_biases)
        {
            tmp3 += displacement;
        }

        if (n_counter == n)
        {
            tmp0 += m;
            tmp2 = out;
        }
    }
    return 0;
}

unsigned char linear_float(unsigned long n, unsigned long d, unsigned long m, unsigned char add_biases, float *multiplier, float *dXm, float *b, float *out)
{
    __m256 operand1 = _mm256_setzero_ps();
    __m256 operand2 = _mm256_setzero_ps();
    __m256 operand3 = _mm256_setzero_ps();
    float *tmp0 = dXm;
    float *tmp1 = multiplier;
    float *tmp2 = out;
    float *tmp3 = b;

    unsigned char displacement;
    unsigned long n_counter = 0;
    unsigned long m_counter = 0;

    for (unsigned int c = 0; c < d * m; c++)
    {
        operand1 = _mm256_loadu_ps(tmp0);
        operand2 = _mm256_loadu_ps(tmp1);
        operand3 = _mm256_loadu_ps(tmp2);
        operand3 = _mm256_fmadd_ps(operand1, operand2, operand3);
        if (add_biases)
        {
            operand1 = _mm256_loadu_ps(tmp3);
            operand3 = _mm256_add_ps(operand1, operand3);
        }
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
            else
            {
                add_biases = 0;
            }
        }
        tmp1 += displacement;
        tmp2 += displacement;
        if (add_biases)
        {
            tmp3 += displacement;
        }
        if (n_counter == n)
        {
            tmp0 += m;
            tmp2 = out;
        }
    }
    return 0;
}

unsigned char squared_loss_float(unsigned long count, float *p_inout, float *t, float *out)
{
    __m256 operand1 = _mm256_setzero_ps();
    __m256 operand2 = _mm256_setzero_ps();
    __m256 operand3 = _mm256_setzero_ps();

    float *tmp0 = p_inout;
    float *tmp1 = t;
    float *tmp2 = out;

    for (unsigned long c = 0; c < count; c += FLOAT_REGISTER_COUNT)
    {
        operand1 = _mm256_loadu_ps(tmp0);
        operand2 = _mm256_loadu_ps(tmp1);
        operand3 = _mm256_sub_ps(operand1, operand2);
        _mm256_storeu_ps(tmp0, operand3);
        operand1 = _mm256_loadu_ps(tmp0);
        _mm256_storeu_ps(tmp0, operand3);
        operand3 = _mm256_mul_ps(operand3, operand3);
        _mm256_storeu_ps(tmp2, operand3);
        tmp0 += FLOAT_REGISTER_COUNT;
        tmp1 += FLOAT_REGISTER_COUNT;
        tmp2 += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char squared_loss_double(unsigned long count, double *p_inout, double *t, double *out)
{
    __m256 operand1 = _mm256_setzero_pd();
    __m256 operand2 = _mm256_setzero_pd();
    __m256 operand3 = _mm256_setzero_pd();

    float *tmp0 = p_inout;
    float *tmp1 = t;
    float *tmp2 = out;

    for (unsigned long c = 0; c < count; c += DOUBLE_REGISTER_COUNT)
    {
        operand1 = _mm256_loadu_pd(tmp0);
        operand2 = _mm256_loadu_pd(tmp1);
        operand3 = _mm256_sub_pd(operand1, operand2); // p - t
        _mm256_storeu_pd(tmp0, operand3);             // p - t
        operand1 = _mm256_loadu_pd(tmp0);
        operand3 = _mm256_mul_pd(operand3, operand3); // pow 2
        _mm256_storeu_pd(tmp2, operand3);
        tmp0 += FLOAT_REGISTER_COUNT;
        tmp1 += FLOAT_REGISTER_COUNT;
        tmp2 += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char scalar_X_matrix_float(unsigned long count, float *m_inout, float *s)
{
    __m256 operand1 = _mm256_setzero_ps();
    __m256 operand2 = _mm256_setzero_ps();

    float *tmp0 = m_inout;
    operand2 = _mm256_loadu_ps(s);

    for (unsigned long c = 0; c < count; c += FLOAT_REGISTER_COUNT)
    {
        operand1 = _mm256_loadu_ps(tmp0);
        operand1 = _mm256_mul_ps(operand1, operand2);
        _mm256_storeu_ps(tmp0, operand1);
        tmp0 += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char scalar_X_matrix_double(unsigned long count, double *m_inout, double *s)
{
    __m256 operand1 = _mm256_setzero_pd();
    __m256 operand2 = _mm256_setzero_pd();

    double *tmp0 = m_inout;
    operand2 = _mm256_loadu_pd(s);

    for (unsigned long c = 0; c < count; c += DOUBLE_REGISTER_COUNT)
    {
        operand1 = _mm256_loadu_pd(tmp0);
        operand1 = _mm256_mul_pd(operand1, operand2);
        _mm256_storeu_pd(tmp0, operand1);
        tmp0 += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}
