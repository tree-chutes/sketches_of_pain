// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>
#include<co5_dl_osx.h>

const unsigned char FLOAT_REGISTER_COUNT = 16;

unsigned char dot_product_float(unsigned long n, unsigned long d, unsigned long m, float *nXd, float *dXm, float *b, float *out)
{
    // The correct calculation will come from finding if d == 1 or not
    // if d != 1 then register reset will come from d. If not then n == m
    // either will work
    unsigned long idx = 0;
    unsigned long register_reset_trigger = (d != 1 ? d : n) / FLOAT_REGISTER_COUNT + ((d != 1 ? d : n) % FLOAT_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change
    unsigned long count = n * m * register_reset_trigger;

    __m512 NxD_s = _mm512_setzero_ps();
    __m512 DxM_s = _mm512_setzero_ps();
    __m512 working_s = _mm512_setzero_ps();

    float *NxD = nXd;
    float *DxM = dXm;
    float *working = malloc(sizeof(float) * FLOAT_REGISTER_COUNT);

    if (working == NULL)
    {
        return 1;
    }

    for (unsigned long c = 0; c < count; c++)
    {
        memset(working, 0, sizeof(float) * FLOAT_REGISTER_COUNT);
        NxD_s = _mm512_loadu_ps(NxD);
        DxM_s = _mm512_loadu_ps(DxM);
        working_s = _mm512_loadu_ps(working);
        working_s = _mm512_fmadd_ps(NxD_s, DxM_s, working_s);
        _mm512_storeu_ps(working, working_s);
        if (d < FLOAT_REGISTER_COUNT || --register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (d < FLOAT_REGISTER_COUNT)
            {
                memset(working + d, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - d));
            }
            else
            {
                memset(working + d % FLOAT_REGISTER_COUNT, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - d % FLOAT_REGISTER_COUNT));
            }
        }
        working_s = _mm512_loadu_ps(working);
        out[idx] += _mm512_reduce_add_ps(working_s);
        // out[idx] += b[(idx % d)];
        if (d < FLOAT_REGISTER_COUNT)
        {
            DxM += d;
            idx++;
        }
        else
        {
            if (register_reset_counter > 0)
            {
                NxD += FLOAT_REGISTER_COUNT;
                DxM += FLOAT_REGISTER_COUNT;
            }
            else
            {
                // They reset in different directions: start of same x row, start next
                // W column
                NxD -= FLOAT_REGISTER_COUNT * (register_reset_trigger - register_reset_counter);
                DxM += d - (FLOAT_REGISTER_COUNT * (register_reset_trigger - register_reset_counter));
                register_reset_counter = register_reset_trigger;
                idx++;
            }
        }
    }
    free(working);
    return 0;
}

unsigned char linear_sgd_float(unsigned long n, unsigned long d, unsigned long m, float *learning_rate, float *weight_gradients, float *weights, float *loss_gradient, float *backpropagating_gradients)
{
    // The correct calculation will come from finding if d == 1 or not
    // if d != 1 then register pointer moves will come from d. If not then n == m
    // either will work

    unsigned long idx = 0;
    unsigned long register_reset_trigger = (d != 1 ? d : n) / FLOAT_REGISTER_COUNT + ((d != 1 ? d : n) % FLOAT_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change
    unsigned long count = d != 1 ? n * m * register_reset_trigger : register_reset_trigger;
    unsigned long offset;

    __m512 weight_gradients_s = _mm512_setzero_ps();
    __m512 weights_s = _mm512_setzero_ps();
    __m512 backpropagating_gradients_s = _mm512_setzero_ps();
    __m512 learning_rate_s = _mm512_loadu_ps(learning_rate);
    __m512 loss_gradient_s = _mm512_loadu_ps(loss_gradient);
    __m512i sign_mask_i = _mm512_set1_epi32(0x80000000);    

    float *weight_gradients_pointer = weight_gradients;
    float *weights_pointer = weights;
    float *loss_gradient_pointer = loss_gradient;
    float *backpropagating_gradients_pointer = backpropagating_gradients;

    for (unsigned long c = 0; c < count; c++)
    {
        weight_gradients_s = _mm512_loadu_ps(weight_gradients_pointer);
        weights_s = _mm512_loadu_ps(weights_pointer);
        backpropagating_gradients_s = _mm512_mul_ps(weights_s, loss_gradient_s);
        weights_s = _mm512_fmsub_ps(weight_gradients_s, learning_rate_s, weights_s);
        weights_s = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(weights_s), sign_mask_i));
        _mm512_storeu_ps(backpropagating_gradients_pointer, backpropagating_gradients_s);
        _mm512_storeu_ps(weights_pointer, weights_s);
        weight_gradients_pointer += FLOAT_REGISTER_COUNT;
        weights_pointer += FLOAT_REGISTER_COUNT;
        backpropagating_gradients_pointer += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char squared_loss_float(unsigned long count, float *prediction, float *truth, float *total)
{
    __m512 prediction_s = _mm512_setzero_ps();
    __m512 truth_s = _mm512_setzero_ps();
    __m512 working_s = _mm512_setzero_ps();

    float *prediction_pointer = prediction;
    float *truth_pointer = truth;
    // REMEMBER vectors are filled when flattened so we can safely advance FLOAT_REGISTER_COUNT
    count = count / FLOAT_REGISTER_COUNT + (count % FLOAT_REGISTER_COUNT != 0);
    for (unsigned long c = 0; c < count; c++)
    {
        prediction_s = _mm512_loadu_ps(prediction_pointer);
        truth_s = _mm512_loadu_ps(truth_pointer);
        working_s = _mm512_sub_ps(prediction_s, truth_s); // p - t
        _mm512_storeu_ps(prediction_pointer, working_s);             // p - t
        prediction_pointer[c] *= 2;                                 // differential
        working_s = _mm512_mul_ps(working_s, working_s); // pow 2
        *total = _mm512_reduce_add_ps(working_s);
        prediction_pointer += FLOAT_REGISTER_COUNT;
        truth_pointer += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char scalar_X_matrix_float(unsigned long count, float *m_inout, float *scalar, float *ret)
{
    __m512 m_inout_s = _mm512_setzero_ps();
    __m512 scalar_s = _mm512_setzero_ps();

    float *m_inout_pointer = m_inout;

    scalar_s = _mm512_loadu_ps(scalar);

    for (unsigned long c = 0; c < count; c += FLOAT_REGISTER_COUNT)
    {
        m_inout_s = _mm512_loadu_ps(m_inout_pointer);
        m_inout_s = _mm512_mul_ps(m_inout_s, scalar_s);
        if (ret == NULL)
        {
            _mm512_storeu_ps(m_inout_pointer, m_inout_s);
        }
        else
        {
            _mm512_storeu_ps(ret, m_inout_s);
            ret += FLOAT_REGISTER_COUNT;
        }
        m_inout_pointer += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char convolve_2d_float(unsigned long w, unsigned long h, unsigned long k, unsigned long o_w, unsigned long count, unsigned short p, unsigned short s, float *hXw, float *kXk, float *previous, float *learning_rate, float *out)
{
    unsigned long idx = 0;
    unsigned long updated_w = w;
    unsigned short k_counter = 0;
    unsigned short w_counter = 0;
    unsigned long register_reset_trigger = k / FLOAT_REGISTER_COUNT + (k % FLOAT_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change

    __m512 feature_s = _mm512_setzero_ps();
    __m512 kernel_s = _mm512_setzero_ps();
    __m512 working_s = _mm512_setzero_ps();

    float *feature_pointer;
    float *updated_feature_pointer = hXw;
    float *kernel_pointer = kXk;
    float *working_pointer = malloc(sizeof(float) * FLOAT_REGISTER_COUNT);

    if (p != 0)
    {
        updated_feature_pointer = pad_matrix_float(h, w, p, hXw);
        updated_w = w + 2 * p;
    }

    if (working_pointer == NULL || updated_feature_pointer == NULL)
    {
        return 1;
    }

    feature_pointer = updated_feature_pointer;
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
            feature_pointer = updated_feature_pointer + (updated_w * w_counter) + s * k_counter;
            kernel_pointer = kXk;
            if (previous != NULL)
            {
                out[idx] = previous[idx] - learning_rate[0] * out[idx];
            }
            idx = c / (k * register_reset_trigger);
            register_reset_counter = register_reset_trigger;
        }

        if (k > FLOAT_REGISTER_COUNT)
        {
            register_reset_counter--;
        }

        memset(working_pointer, 0, sizeof(float) * FLOAT_REGISTER_COUNT);
        feature_s = _mm512_loadu_ps(feature_pointer);
        kernel_s = _mm512_loadu_ps(kernel_pointer);
        working_s = _mm512_loadu_ps(working_pointer);
        working_s = _mm512_fmadd_ps(feature_s, kernel_s, working_s);
        _mm512_storeu_ps(working_pointer, working_s);
        if (k < FLOAT_REGISTER_COUNT || register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (k < FLOAT_REGISTER_COUNT)
            {
                memset(working_pointer + k, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - k));
            }
            else
            {
                memset(working_pointer + k % FLOAT_REGISTER_COUNT, 0, sizeof(float) * (FLOAT_REGISTER_COUNT - k % FLOAT_REGISTER_COUNT));
            }
        }
        working_s = _mm512_loadu_ps(working_pointer);
        out[idx] += _mm512_reduce_add_ps(working_s);

        if (k < FLOAT_REGISTER_COUNT)
        {
            kernel_pointer += k;
            feature_pointer += updated_w;
        }
        else
        {
            if (register_reset_counter > 0)
            {
                feature_pointer += FLOAT_REGISTER_COUNT;
                kernel_pointer += FLOAT_REGISTER_COUNT;
            }
            else
            {
                // They reset same direction beginning of next row
                feature_pointer += updated_w - k + 1;
                kernel_pointer += k % FLOAT_REGISTER_COUNT;
                register_reset_counter = register_reset_trigger;
            }
        }
    }
    // last idx ALREADY set up in loop
    if (previous != NULL)
    {
        out[idx] = previous[idx] - learning_rate[0] * out[idx];
    }
    free(working_pointer);
    if (p != 0)
    {
        free(updated_feature_pointer);
    }
    return 0;
}

float* pad_matrix_float(unsigned long h, unsigned long w, unsigned short p, float *m)
{
    float *start;
    unsigned long padded_width = w + 2 * p;
    unsigned long padded_height = h + 2 * p;
    unsigned long buffer_size = padded_height * padded_width;
    float *ret = malloc(sizeof(float) * buffer_size);

    if (ret != NULL)
    {
        start = ret + p * padded_width;
        memset(ret, 0, buffer_size);

        for (unsigned long i = 0; i < h; i++)
        {
            start += p; // padded head
            memcpy(start, m + (i * w), sizeof(float) * w);
            start += w + p; // w + padded tail
        }
    }
    return ret;
}
