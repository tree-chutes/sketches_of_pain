// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>

const unsigned char FLOAT_REGISTER_COUNT = 16;

unsigned char dot_product_float(unsigned long n, unsigned long d, unsigned long m, float *nXd, float *dXm, float *b, float *out)
{
    unsigned long idx = 0;
    unsigned long register_bumps = 0; //This should disappear with detailed analysis
    unsigned long register_reset_trigger = d / FLOAT_REGISTER_COUNT + (d % FLOAT_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; //keeps track of when the pointers need to change
    unsigned long count = n * m * register_reset_trigger;

    float *working = malloc(sizeof(float) * FLOAT_REGISTER_COUNT);    
    __m512 operand1 = _mm512_setzero_ps();
    __m512 operand2 = _mm512_setzero_ps();
    __m512 operand3 = _mm512_setzero_ps();
    float *NxD = nXd;
    float *DxM = dXm;

    for (unsigned long c = 0; c < count; c++)
    {
        if (d < FLOAT_REGISTER_COUNT)
        {
            idx = c;  
        }
        else
        {
            register_reset_counter--;
        }
        memset(working, 0, sizeof(float) * FLOAT_REGISTER_COUNT);
        operand1 = _mm512_loadu_ps(NxD);
        operand2 = _mm512_loadu_ps(DxM);
        operand3 = _mm512_loadu_ps(working);
        operand3 = _mm512_fmadd_ps(operand1, operand2, operand3);
        _mm512_storeu_ps(working, operand3);
        if (d < FLOAT_REGISTER_COUNT || register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (d < FLOAT_REGISTER_COUNT)
            {
                memset(working + d , 0, sizeof(float) * (FLOAT_REGISTER_COUNT -  d)); 
            }
            else
            {
                memset(working + d % FLOAT_REGISTER_COUNT, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - d % FLOAT_REGISTER_COUNT) ); 
            }
        }
        operand3 = _mm512_loadu_ps(working);
        out[idx] += _mm512_reduce_add_ps(operand3);
        out[idx] += b[(idx % d)];
        if ((idx + 1) % m == 0)
        {
            if (d > FLOAT_REGISTER_COUNT)
            {
                NxD -= (FLOAT_REGISTER_COUNT * register_bumps);
                register_reset_counter = register_reset_trigger;
                idx++;
                register_bumps = 0;
            }
            NxD += d;
            DxM = dXm;
        }
        else
        {
            if (d < FLOAT_REGISTER_COUNT)
            {
                DxM += d;
            }
            else
            {
                if (register_reset_counter > 0)
                {
                    register_bumps++;
                    NxD += FLOAT_REGISTER_COUNT;
                    DxM += FLOAT_REGISTER_COUNT;
                }
                else
                {
                    //They reset in different directions: start of same x row, start next 
                    //W column 
                    NxD -= (FLOAT_REGISTER_COUNT * register_bumps);
                    DxM += d - (FLOAT_REGISTER_COUNT * register_bumps);
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

unsigned char sgd_float(unsigned long n, unsigned long d, unsigned long m, float *learning_rate, float *nXd, float *dXm, float *previous, float *out)
{
    unsigned long idx = 0;
    unsigned long offset;
    unsigned long register_bumps = 0; //This should disappear with detailed analysis
    unsigned long register_reset_trigger = d / FLOAT_REGISTER_COUNT + (d % FLOAT_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; //keeps track of when the pointers need to change
    unsigned long count = n * m * register_reset_trigger;

    float *working = malloc(sizeof(float) * FLOAT_REGISTER_COUNT);
    __m512 operand1 = _mm512_setzero_ps();
    __m512 operand2 = _mm512_setzero_ps();
    __m512 operand3 = _mm512_setzero_ps();
    float *NxD = nXd;
    float *DxM = dXm;

    for (unsigned long c = 0; c < count; c++)
    {
        if (d < FLOAT_REGISTER_COUNT)
        {
            idx = c;
        }
        else
        {
            register_reset_counter--;
        }
        memset(working, 0, sizeof(float) * FLOAT_REGISTER_COUNT);        
        operand1 = _mm512_loadu_ps(NxD);
        operand2 = _mm512_loadu_ps(DxM);
        operand3 = _mm512_loadu_ps(working);
        operand3 = _mm512_fmadd_ps(operand1, operand2, operand3);
        _mm512_storeu_ps(working, operand3);
        if (d < FLOAT_REGISTER_COUNT || register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (d < FLOAT_REGISTER_COUNT)
            {
                memset(working + d , 0, sizeof(float) * (FLOAT_REGISTER_COUNT -  d)); 
            }
            else
            {
                memset(working + d % FLOAT_REGISTER_COUNT, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - d % FLOAT_REGISTER_COUNT) ); 
            }
        }
        operand3 = _mm512_loadu_ps(working);
        out[idx] += _mm512_reduce_add_ps(operand3);
        // OR takes care of D > FLOAT_REGISTER_COUNT  
        if ((idx + 1) % m == 0 && d < FLOAT_REGISTER_COUNT || (idx + 1) % m == 0 && register_reset_counter == 0)
        {
            offset = ((idx + 1) / m  - 1) * m;
            out[idx] = previous[idx] - learning_rate[0] * out[idx];

            // CLEARS fill from out buffer for carrying over purposes. Otherwise working buffer contains
            // garbage values for the following idx
            if (d < FLOAT_REGISTER_COUNT)
            {
                memset(out + offset + d, 0, sizeof(float) * (FLOAT_REGISTER_COUNT -  d)); 
            }
            else
            {
                memset(out + offset + d, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - d % FLOAT_REGISTER_COUNT) ); 
                NxD -= (FLOAT_REGISTER_COUNT * register_bumps);
                register_reset_counter = register_reset_trigger;
                idx++;
                register_bumps = 0;

            }
            NxD += d;
            DxM = dXm;
        }
        else
        {
            if (d < FLOAT_REGISTER_COUNT)
            {
                out[idx] = previous[idx] - learning_rate[0] * out[idx];
                DxM += d;
            }
            else
            {
                if (register_reset_counter > 0)
                {
                    register_bumps++;
                    NxD += FLOAT_REGISTER_COUNT;
                    DxM += FLOAT_REGISTER_COUNT;
                }
                else
                {
                    //They reset in different directions: start of same x row, start next 
                    //W column 
                    NxD -= (FLOAT_REGISTER_COUNT * register_bumps);
                    DxM += d - (FLOAT_REGISTER_COUNT * register_bumps);
                    register_reset_counter = register_reset_trigger;
                    out[idx] = previous[idx] - learning_rate[0] * out[idx];
                    idx++;
                    register_bumps = 0;
                }                    
            }
        }
    }
    free(working);
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
        operand3 = _mm512_sub_ps(operand1, operand2); // p - t
        _mm512_storeu_ps(tmp0, operand3);             // p - t
        operand1 = _mm512_loadu_ps(tmp0);
        operand3 = _mm512_mul_ps(operand3, operand3); // pow 2
        _mm512_storeu_ps(tmp2, operand3);
        tmp0 += FLOAT_REGISTER_COUNT;
        tmp1 += FLOAT_REGISTER_COUNT;
        tmp2 += FLOAT_REGISTER_COUNT;
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
