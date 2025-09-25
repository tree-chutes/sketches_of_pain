// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>

const unsigned char FLOAT_REGISTER_COUNT = 16;

unsigned char linear_float(unsigned long n, unsigned long d, unsigned long m, float *nXd, float *dXm, float *b, float *out)
{
    float *working = malloc(sizeof(float) * FLOAT_REGISTER_COUNT);    
    __m512 operand1 = _mm512_setzero_ps();
    __m512 operand2 = _mm512_setzero_ps();
    __m512 operand3 = _mm512_setzero_ps();
    float *tmp0 = nXd;
    float *tmp1 = dXm;
    float *tmp2 = working;

    for (unsigned long c = 0; c < n * m; c++)
    {
        memset(working, 0, sizeof(float) * FLOAT_REGISTER_COUNT);
        operand1 = _mm512_loadu_ps(tmp0);
        operand2 = _mm512_loadu_ps(tmp1);
        operand3 = _mm512_loadu_ps(tmp2);
        operand3 = _mm512_fmadd_ps(operand1, operand2, operand3);
        _mm512_storeu_ps(tmp2, operand3);
        memset(tmp2 + d, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - d));
        operand3 = _mm512_loadu_ps(tmp2);
        out[c] = _mm512_reduce_add_ps(operand3);
        if (b != NULL)
        {
            out[c] += b[c];
        }
        tmp1 += d;
        if ((c + 1) % m == 0)
        {
            tmp0 += d;
            tmp1 = dXm;
        }
    }
    free(working);
    return 0;
}

unsigned char derivative_linear_float(unsigned long n, unsigned long d, unsigned long m, float *learning_rate, float *nXd, float *dXm, float *previous, float *working_vec)
{
    __m512d operand1 = _mm512_setzero_ps();
    __m512d operand2 = _mm512_setzero_ps();
    __m512d operand3 = _mm512_setzero_ps();
    __m512d operand4 = _mm512_setzero_ps();

    float *tmp0 = dXm;
    float *tmp1 = nXd;
    float *tmp2 = working_vec;
    float *tmp3 = previous;
    operand4 = _mm512_loadu_ps(learning_rate);

    for (unsigned int c = 0; c < n * m; c++)
    {
        operand1 = _mm512_loadu_ps(tmp0);
        operand2 = _mm512_loadu_ps(tmp1);
        operand3 = _mm512_loadu_ps(tmp2);
        operand3 = _mm512_fmadd_ps(operand1, operand2, operand3);
        tmp0 += m;
        tmp1 += m;
        if ((c + 1) % m == 0)
        {
            operand1 = _mm512_loadu_ps(tmp3);
            operand2 = _mm512_mul_ps(operand4, operand3);
            operand1 = _mm512_sub_ps(operand1, operand2);
            _mm512_storeu_ps(tmp2, operand1);
            tmp2 += m;
            tmp3 += m;
            memset(tmp2, 0, sizeof(float) * n * (m - (c + 1) % m));
            tmp1 = dXm;
        }
        else
        {
            _mm512_storeu_ps(tmp2, operand3);
        }
    }
    return 0;
}

unsigned char scalar_X_matrix_float(unsigned long count, float *m_inout, float *s)
{
    __m512 operand1 = _mm512_setzero_ps();
    __m512 operand2 = _mm512_setzero_ps();

    float *tmp0 = m_inout;
    operand2 = _mm512_loadu_ps(s);

    for (unsigned long c = 0; c < count; c += FLOAT_REGISTER_COUNT)
    {
        operand1 = _mm512_loadu_ps(tmp0);
        operand1 = _mm512_mul_ps(operand1, operand2);
        _mm512_storeu_ps(tmp0, operand1);
        tmp0 += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char squared_loss_float(unsigned long count, float *p_inout, float *t, float *out)
{
    __m512 operand1 = _mm512_setzero_ps();
    __m512 operand2 = _mm512_setzero_ps();
    __m512 operand3 = _mm512_setzero_ps();

    float *tmp0 = p_inout;
    float *tmp1 = t;
    float *tmp2 = out;

    for (unsigned long c = 0; c < count; c += FLOAT_REGISTER_COUNT)
    {
        operand1 = _mm512_loadu_ps(tmp0);
        operand2 = _mm512_loadu_ps(tmp1);
        operand3 = _mm512_sub_ps(operand1, operand2);
        _mm512_storeu_ps(tmp0, operand3);
        operand1 = _mm512_loadu_ps(tmp0);
        _mm512_storeu_ps(tmp0, operand3);
        operand3 = _mm512_mul_ps(operand3, operand3);
        _mm512_storeu_ps(tmp2, operand3);
        tmp0 += FLOAT_REGISTER_COUNT;
        tmp1 += FLOAT_REGISTER_COUNT;
        tmp2 += FLOAT_REGISTER_COUNT;
    }
    return 0;
}
