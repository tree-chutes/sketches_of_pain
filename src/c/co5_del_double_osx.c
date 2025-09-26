// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>

const unsigned char DOUBLE_REGISTER_COUNT = 8;

unsigned char dot_product_double(unsigned long n, unsigned long d, unsigned long m, double *nXd, double *dXm, double *b, double *out)
{
    double*working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);    
    __m512 operand1 = _mm512_setzero_pd();
    __m512 operand2 = _mm512_setzero_pd();
    __m512 operand3 = _mm512_setzero_pd();
    double*tmp0 = nXd;
    double*tmp1 = dXm;
    double*tmp2 = working;

    for (unsigned long c = 0; c < n * m; c++)
    {
        memset(working, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        operand1 = _mm512_loadu_pd(tmp0);
        operand2 = _mm512_loadu_pd(tmp1);
        operand3 = _mm512_loadu_pd(tmp2);
        operand3 = _mm512_fmadd_pd(operand1, operand2, operand3);
        _mm512_storeu_pd(tmp2, operand3);
        memset(tmp2 + d, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - d));
        operand3 = _mm512_loadu_pd(tmp2);
        out[c] = _mm512_reduce_add_pd(operand3);
        out[c] += b[c];
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

unsigned char differentiate_double(unsigned long n, unsigned long d, unsigned long m, double *learning_rate, double *nXd, double *dXm, double *previous, double *out)
{
    unsigned long offset;
    double *working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);    
    __m512 operand1 = _mm512_setzero_pd();
    __m512 operand2 = _mm512_setzero_pd();
    __m512 operand3 = _mm512_setzero_pd();
    double *tmp0 = nXd;
    double *tmp1 = dXm;
    double *tmp2 = working;

    for (unsigned long c = 0; c < n * m; c++)
    {
        memset(working, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        operand1 = _mm512_loadu_pd(tmp0);
        operand2 = _mm512_loadu_pd(tmp1);
        operand3 = _mm512_loadu_pd(tmp2);
        operand3 = _mm512_fmadd_pd(operand1, operand2, operand3);
        _mm512_storeu_pd(tmp2, operand3);
        memset(tmp2 + d, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - d));
        operand3 = _mm512_loadu_pd(tmp2);
        out[c] = _mm512_reduce_add_pd(operand3);
        tmp1 += d;
        if ((c + 1) % m == 0)
        {
            offset = ((c + 1) / m  - 1) * m;
            operand1 = _mm512_loadu_pd(out + offset);
            operand2 = _mm512_loadu_pd(learning_rate + offset);
            operand3 = _mm512_loadu_pd(previous + offset);
            operand2 = _mm512_mul_pd(operand1, operand2);
            operand3 = _mm512_sub_pd(operand3, operand2);
            _mm512_storeu_pd(out + offset, operand3);
            tmp0 += d;
            tmp1 = dXm;
        }
    }
    free(working);
    return 0;
}


unsigned char squared_loss_double(unsigned long count, double *p_inout, double *t, double *out)
{
    __m512 operand1 = _mm512_setzero_pd();
    __m512 operand2 = _mm512_setzero_pd();
    __m512 operand3 = _mm512_setzero_pd();

    double *tmp0 = p_inout;
    double *tmp1 = t;
    double *tmp2 = out;

    for (unsigned long c = 0; c < count; c += DOUBLE_REGISTER_COUNT)
    {
        operand1 = _mm512_loadu_pd(tmp0);
        operand2 = _mm512_loadu_pd(tmp1);
        operand3 = _mm512_sub_pd(operand1, operand2); // p - t
        _mm512_storeu_pd(tmp0, operand3);             // p - t
        operand1 = _mm512_loadu_pd(tmp0);
        operand3 = _mm512_mul_pd(operand3, operand3); // pow 2
        _mm512_storeu_pd(tmp2, operand3);
        tmp0 += DOUBLE_REGISTER_COUNT;
        tmp1 += DOUBLE_REGISTER_COUNT;
        tmp2 += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}

unsigned char scalar_X_matrix_double(unsigned long count, double *m_inout, double *s)
{
    __m512 operand1 = _mm512_setzero_pd();
    __m512 operand2 = _mm512_setzero_pd();

    double *tmp0 = m_inout;
    operand2 = _mm512_loadu_pd(s);

    for (unsigned long c = 0; c < count; c += DOUBLE_REGISTER_COUNT)
    {
        operand1 = _mm512_loadu_pd(tmp0);
        operand1 = _mm512_mul_pd(operand1, operand2);
        _mm512_storeu_pd(tmp0, operand1);
        tmp0 += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}
