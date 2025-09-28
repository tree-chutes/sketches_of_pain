// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>

const unsigned char DOUBLE_REGISTER_COUNT = 8;

unsigned char dot_product_double(unsigned long n, unsigned long d, unsigned long m, double *nXd, double *dXm, double *b, double *out)
{
    unsigned long idx = 0;
    unsigned long register_bumps = 0; //This should disappear with detailed analysis
    unsigned long register_reset_trigger = d / DOUBLE_REGISTER_COUNT + (d % DOUBLE_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; //keeps track of when the pointers need to change
    unsigned long count = n * n * register_reset_trigger;

    double *working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);    
    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();
    double *tmp0 = nXd;
    double *tmp1 = dXm;


    for (unsigned long c = 0; c < count; c++)
    {
        if (d < DOUBLE_REGISTER_COUNT)
        {
            idx = c;  
        }
        else
        {
            register_reset_counter--;
        }
        memset(working, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        operand1 = _mm512_loadu_pd(tmp0);
        operand2 = _mm512_loadu_pd(tmp1);
        operand3 = _mm512_loadu_pd(working);
        operand3 = _mm512_fmadd_pd(operand1, operand2, operand3);
        _mm512_storeu_pd(working, operand3);
        if (d < DOUBLE_REGISTER_COUNT || register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (d < DOUBLE_REGISTER_COUNT)
            {
                memset(working + d , 0, sizeof(double) * (DOUBLE_REGISTER_COUNT -  d)); 
            }
            else
            {
                memset(working + d % DOUBLE_REGISTER_COUNT, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - d % DOUBLE_REGISTER_COUNT) ); 
            }
        }
        operand3 = _mm512_loadu_pd(working);
        out[idx] += _mm512_reduce_add_pd(operand3);
        out[idx] += b[(idx % d)];
        if ((idx + 1) % m == 0 && register_reset_counter == 0)
        {
            if (d > DOUBLE_REGISTER_COUNT)
            {
                tmp0 -= (DOUBLE_REGISTER_COUNT * register_bumps);
                register_reset_counter = register_reset_trigger;
                idx++;
                register_bumps = 0;
            }
            tmp0 += d;
            tmp1 = dXm;
        }
        else
        {
            if (d < DOUBLE_REGISTER_COUNT)
            {
                tmp1 += d;
            }
            else
            {
                if (register_reset_counter > 0)
                {
                    register_bumps++;
                    tmp0 += DOUBLE_REGISTER_COUNT;
                    tmp1 += DOUBLE_REGISTER_COUNT;
                }
                else
                {
                    //They reset in different directions: start of same x row, start next 
                    //W column 
                    tmp0 -= (DOUBLE_REGISTER_COUNT * register_bumps);
                    tmp1 += d - (DOUBLE_REGISTER_COUNT * register_bumps);
                    register_reset_counter = register_reset_trigger;
                    idx++;
                    register_bumps = 0;
                }                    
            }
        }
    }
    free(working);
    return 0;
}

unsigned char differentiate_double(unsigned long n, unsigned long d, unsigned long m, double *learning_rate, double *nXd, double *dXm, double *previous, double *out)
{
    unsigned long offset;
    unsigned long count = n * m;
    double *working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);    
    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();
    double *tmp0 = nXd;
    double *tmp1 = dXm;
    double *tmp2 = working;

    for (unsigned long c = 0; c < count; c++)
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
    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();

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
    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();

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
