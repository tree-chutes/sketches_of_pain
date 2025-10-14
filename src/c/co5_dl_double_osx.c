// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>
#include <co5_dl_osx.h>

const unsigned char DOUBLE_REGISTER_COUNT = 8;

unsigned char dot_product_double(unsigned long n, unsigned long d, unsigned long m, double *nXd, double *dXm, double *b, double *out)
{
    // The correct calculation will come from finding if d == 1 or not
    // if d != 1 then register reset will come from d. If not then n == m
    // either will work
    unsigned long idx = 0;
    unsigned long register_reset_trigger = (d != 1 ? d : n) / DOUBLE_REGISTER_COUNT + ((d != 1 ? d : n) % DOUBLE_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change
    unsigned long count = n * m * register_reset_trigger;

    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();

    double *NxD = nXd;
    double *DxM = dXm;
    double *working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);

    if (working == NULL)
    {
        return 1;
    }

    for (unsigned long c = 0; c < count; c++)
    {
        memset(working, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        operand1 = _mm512_loadu_pd(NxD);
        operand2 = _mm512_loadu_pd(DxM);
        operand3 = _mm512_loadu_pd(working);
        operand3 = _mm512_fmadd_pd(operand1, operand2, operand3);
        _mm512_storeu_pd(working, operand3);
        if (d < DOUBLE_REGISTER_COUNT || --register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (d < DOUBLE_REGISTER_COUNT)
            {
                memset(working + d, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - d));
            }
            else
            {
                memset(working + d % DOUBLE_REGISTER_COUNT, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - d % DOUBLE_REGISTER_COUNT));
            }
        }
        operand3 = _mm512_loadu_pd(working);
        out[idx] += _mm512_reduce_add_pd(operand3);
        out[idx] += b[(idx % d)];
        if (d < DOUBLE_REGISTER_COUNT)
        {
            DxM += d;
            idx++;
        }
        else
        {
            if (register_reset_counter > 0)
            {
                NxD += DOUBLE_REGISTER_COUNT;
                DxM += DOUBLE_REGISTER_COUNT;
            }
            else
            {
                // They reset in different directions: start of same x row, start next
                // W column
                NxD -= DOUBLE_REGISTER_COUNT * (register_reset_trigger - register_reset_counter);
                DxM += d - (DOUBLE_REGISTER_COUNT * (register_reset_trigger - register_reset_counter));
                register_reset_counter = register_reset_trigger;
                idx++;
            }
        }
    }
    free(working);
    return 0;
}

unsigned char linear_sgd_double(unsigned long n, unsigned long d, unsigned long m, double *learning_rate, double *y, double *derivative, double *previous, double *out)
{
    // The correct calculation will come from finding if d == 1 or not
    // if d != 1 then register pointer moves will come from d. If not then n == m
    // either will work

    unsigned long idx = 0;
    unsigned long register_reset_trigger = (d != 1 ? d : n) / DOUBLE_REGISTER_COUNT + ((d != 1 ? d : n) % DOUBLE_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change
    unsigned long count = d != 1 ? n * m * register_reset_trigger : register_reset_trigger;
    unsigned long offset;

    __m512d predicted_s = _mm512_setzero_pd();
    __m512d differential_s = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();
    __m512d learning_rate_s = _mm512_loadu_pd(learning_rate);
    double *predicted = y;
    double *differential = derivative;
    double *working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);

    if (working == NULL)
    {
        return 1;
    }

    for (unsigned long c = 0; c < count; c++)
    {
        memset(working, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        predicted_s = _mm512_loadu_pd(predicted);
        differential_s = _mm512_loadu_pd(differential);
        operand3 = _mm512_loadu_pd(working);
        operand3 = _mm512_mul_pd(learning_rate_s, differential_s);
        operand3 = _mm512_sub_pd(predicted_s, operand3);
        _mm512_storeu_pd(out, operand3);
        predicted += DOUBLE_REGISTER_COUNT;
        differential += DOUBLE_REGISTER_COUNT;
        out += DOUBLE_REGISTER_COUNT;
    }
    free(working);
    return 0;
}

unsigned char squared_loss_double(unsigned long count, double *p, double *t, double *total)
{
    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();

    double *tmp0 = p;
    double *tmp1 = t;
    // REMEMBER vectors are filled when flattened so can safely advance DOUBLE_REGISTER_COUNT
    count = count / DOUBLE_REGISTER_COUNT + (count % DOUBLE_REGISTER_COUNT != 0);
    for (unsigned long c = 0; c < count; c++)
    {
        operand1 = _mm512_loadu_pd(tmp0);
        operand2 = _mm512_loadu_pd(tmp1);
        operand3 = _mm512_sub_pd(operand1, operand2); // p - t
        _mm512_storeu_pd(tmp0, operand3);             // p - t
        tmp0[c] *= 2;                                 // differential
        operand3 = _mm512_mul_pd(operand3, operand3); // pow 2
        *total = _mm512_reduce_add_pd(operand3);
        tmp0 += DOUBLE_REGISTER_COUNT;
        tmp1 += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}

unsigned char scalar_X_matrix_double(unsigned long count, double *m_inout, double *s, double *ret)
{
    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();

    double *tmp0 = m_inout;
    operand2 = _mm512_loadu_pd(s);

    for (unsigned long c = 0; c < count; c += DOUBLE_REGISTER_COUNT)
    {
        operand1 = _mm512_loadu_pd(tmp0);
        operand1 = _mm512_mul_pd(operand1, operand2);
        if (ret == NULL)
        {
            _mm512_storeu_pd(tmp0, operand1);
        }
        else
        {
            _mm512_storeu_pd(ret, operand1);
            ret += DOUBLE_REGISTER_COUNT;
        }
        tmp0 += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}

unsigned char convolve_2d_double(unsigned long w, unsigned long h, unsigned long k, unsigned long o_w, unsigned long count, unsigned short p, unsigned short s, double *hXw, double *kXk, double *previous, double *learning_rate, double *out)
{
    unsigned long idx = 0;
    unsigned long updated_w = w;
    unsigned short k_counter = 0;
    unsigned short w_counter = 0;
    unsigned long register_reset_trigger = k / DOUBLE_REGISTER_COUNT + (k % DOUBLE_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change

    __m512d operand1 = _mm512_setzero_pd();
    __m512d operand2 = _mm512_setzero_pd();
    __m512d operand3 = _mm512_setzero_pd();

    double *HxW;
    double *updated_hXw = hXw;
    double *KxK = kXk;
    double *working = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);

    if (p != 0)
    {
        updated_hXw = pad_matrix_double(h, w, p, hXw);
        updated_w = w + 2 * p;
    }

    if (working == NULL || updated_hXw == NULL)
    {
        return 1;
    }

    HxW = updated_hXw;
    count *= k * register_reset_trigger;
    for (unsigned long c = 0; c < count; c++)
    {
        if (c > 0 && c % (k * register_reset_trigger) == 0)
        {
            k_counter++;
            if (k_counter == o_w)
            {
                w_counter++;
                k_counter = 0;
            }
            HxW = updated_hXw + (updated_w * w_counter) + s * k_counter;
            KxK = kXk;
            if (previous != NULL)
            {
                out[idx] = previous[idx] - learning_rate[0] * out[idx];
            }
            idx = c / (k * register_reset_trigger);
            register_reset_counter = register_reset_trigger;
        }

        if (k > DOUBLE_REGISTER_COUNT)
        {
            register_reset_counter--;
        }

        memset(working, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        operand1 = _mm512_loadu_pd(HxW);
        operand2 = _mm512_loadu_pd(KxK);
        operand3 = _mm512_loadu_pd(working);
        operand3 = _mm512_fmadd_pd(operand1, operand2, operand3);
        _mm512_storeu_pd(working, operand3);
        if (k < DOUBLE_REGISTER_COUNT || register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (k < DOUBLE_REGISTER_COUNT)
            {
                memset(working + k, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - k));
            }
            else
            {
                memset(working + k % DOUBLE_REGISTER_COUNT, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - k % DOUBLE_REGISTER_COUNT));
            }
        }
        operand3 = _mm512_loadu_pd(working);
        out[idx] += _mm512_reduce_add_pd(operand3);

        if (k < DOUBLE_REGISTER_COUNT)
        {
            KxK += k;
            HxW += updated_w;
        }
        else
        {
            if (register_reset_counter > 0)
            {
                HxW += DOUBLE_REGISTER_COUNT;
                KxK += DOUBLE_REGISTER_COUNT;
            }
            else
            {
                // They reset same direction beginning of next row
                HxW += updated_w - k + 1;
                KxK += k % DOUBLE_REGISTER_COUNT;
                register_reset_counter = register_reset_trigger;
            }
        }
    }
    //last idx ALREADY set up in loop
    if (previous != NULL)
    {
        out[idx] = previous[idx] - learning_rate[0] * out[idx];
    }
    free(working);
    if (p != 0)
    {
        free(updated_hXw);
    }
    return 0;
}

double *pad_matrix_double(unsigned long h, unsigned long w, unsigned short p, double *m)
{
    double *start;
    unsigned long padded_width = w + 2 * p;
    unsigned long padded_height = h + 2 * p;
    unsigned long buffer_size = padded_height * padded_width;
    double *ret = malloc(sizeof(double) * buffer_size);

    if (ret != NULL)
    {
        start = ret + p * padded_width;
        memset(ret, 0, buffer_size);

        for (unsigned long i = 0; i < h; i++)
        {
            start += p; // padded head
            memcpy(start, m + (i * w), sizeof(double) * w);
            start += w + p; // w + padded tail
        }
    }
    return ret;
}
