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

    __m512d NxD_s = _mm512_setzero_pd();
    __m512d DxM_s = _mm512_setzero_pd();
    __m512d working_s = _mm512_setzero_pd();

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
        NxD_s = _mm512_loadu_pd(NxD);
        DxM_s = _mm512_loadu_pd(DxM);
        working_s = _mm512_loadu_pd(working);
        working_s = _mm512_fmadd_pd(NxD_s, DxM_s, working_s);
        _mm512_storeu_pd(working, working_s);
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
        working_s = _mm512_loadu_pd(working);
        out[idx] += _mm512_reduce_add_pd(working_s);
        // out[idx] += b[(idx % d)];
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

unsigned char linear_sgd_double(unsigned long n, unsigned long d, unsigned long m, double *learning_rate, double *weight_gradients, double *weights, double *loss_gradient, double *backpropagating_gradients)
{
    // The correct calculation will come from finding if d == 1 or not
    // if d != 1 then register pointer moves will come from d. If not then n == m
    // either will work

    unsigned long idx = 0;
    unsigned long register_reset_trigger = (d != 1 ? d : n) / DOUBLE_REGISTER_COUNT + ((d != 1 ? d : n) % DOUBLE_REGISTER_COUNT != 0);
    unsigned long register_reset_counter = register_reset_trigger; // keeps track of when the pointers need to change
    unsigned long count = d != 1 ? n * m * register_reset_trigger : register_reset_trigger;
    unsigned long offset;

    __m512d weight_gradients_s = _mm512_setzero_pd();
    __m512d weights_s = _mm512_setzero_pd();
    __m512d backpropagating_gradients_s = _mm512_setzero_pd();
    __m512d learning_rate_s = _mm512_loadu_pd(learning_rate);
    __m512d loss_gradient_s = _mm512_loadu_pd(loss_gradient);
    __m512i sign_mask_i = _mm512_set1_epi64(0x8000000000000000ULL);    

    double *weight_gradients_pointer = weight_gradients;
    double *weights_pointer = weights;
    double *loss_gradient_pointer = loss_gradient;
    double *backpropagating_gradients_pointer = backpropagating_gradients;

    for (unsigned long c = 0; c < count; c++)
    {
        weight_gradients_s = _mm512_loadu_pd(weight_gradients_pointer);
        weights_s = _mm512_loadu_pd(weights_pointer);
        backpropagating_gradients_s = _mm512_mul_pd(weights_s, loss_gradient_s);
        weights_s = _mm512_fmsub_pd(weight_gradients_s, learning_rate_s, weights_s);
        weights_s = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(weights_s), sign_mask_i));
        _mm512_storeu_pd(backpropagating_gradients_pointer, backpropagating_gradients_s);
        _mm512_storeu_pd(weights_pointer, weights_s);
        weight_gradients_pointer += DOUBLE_REGISTER_COUNT;
        weights_pointer += DOUBLE_REGISTER_COUNT;
        backpropagating_gradients_pointer += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}

unsigned char squared_loss_double(unsigned long count, double *prediction, double *truth, double *total)
{
    __m512d prediction_s = _mm512_setzero_pd();
    __m512d truth_s = _mm512_setzero_pd();
    __m512d working_s = _mm512_setzero_pd();

    double *prediction_pointer = prediction;
    double *truth_pointer = truth;
    // REMEMBER vectors are filled when flattened so we can safely advance DOUBLE_REGISTER_COUNT
    count = count / DOUBLE_REGISTER_COUNT + (count % DOUBLE_REGISTER_COUNT != 0);
    for (unsigned long c = 0; c < count; c++)
    {
        prediction_s = _mm512_loadu_pd(prediction_pointer);
        truth_s = _mm512_loadu_pd(truth_pointer);
        working_s = _mm512_sub_pd(prediction_s, truth_s); // p - t
        _mm512_storeu_pd(prediction_pointer, working_s);             // p - t
        prediction_pointer[c] *= 2;                                 // differential
        working_s = _mm512_mul_pd(working_s, working_s); // pow 2
        *total = _mm512_reduce_add_pd(working_s);
        prediction_pointer += DOUBLE_REGISTER_COUNT;
        truth_pointer += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}

unsigned char scalar_X_matrix_double(unsigned long count, double *m_inout, double *scalar, double *ret)
{
    __m512d m_inout_s = _mm512_setzero_pd();
    __m512d scalar_s = _mm512_setzero_pd();

    double *m_inout_pointer = m_inout;

    scalar_s = _mm512_loadu_pd(scalar);

    for (unsigned long c = 0; c < count; c += DOUBLE_REGISTER_COUNT)
    {
        m_inout_s = _mm512_loadu_pd(m_inout_pointer);
        m_inout_s = _mm512_mul_pd(m_inout_s, scalar_s);
        if (ret == NULL)
        {
            _mm512_storeu_pd(m_inout_pointer, m_inout_s);
        }
        else
        {
            _mm512_storeu_pd(ret, m_inout_s);
            ret += DOUBLE_REGISTER_COUNT;
        }
        m_inout_pointer += DOUBLE_REGISTER_COUNT;
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

    __m512d feature_s = _mm512_setzero_pd();
    __m512d kernel_s = _mm512_setzero_pd();
    __m512d working_s = _mm512_setzero_pd();

    double *feature_pointer;
    double *updated_feature_pointer = hXw;
    double *kernel_pointer = kXk;
    double *working_pointer = malloc(sizeof(double) * DOUBLE_REGISTER_COUNT);

    if (p != 0)
    {
        updated_feature_pointer = pad_matrix_double(h, w, p, hXw);
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

        if (k > DOUBLE_REGISTER_COUNT)
        {
            register_reset_counter--;
        }

        memset(working_pointer, 0, sizeof(double) * DOUBLE_REGISTER_COUNT);
        feature_s = _mm512_loadu_pd(feature_pointer);
        kernel_s = _mm512_loadu_pd(kernel_pointer);
        working_s = _mm512_loadu_pd(working_pointer);
        working_s = _mm512_fmadd_pd(feature_s, kernel_s, working_s);
        _mm512_storeu_pd(working_pointer, working_s);
        if (k < DOUBLE_REGISTER_COUNT || register_reset_counter == 0)
        {
            // CLEARS fill from working buffer for reducing purposes. Otherwise working buffer contains
            // garbage values for the current idx
            if (k < DOUBLE_REGISTER_COUNT)
            {
                memset(working_pointer + k, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - k));
            }
            else
            {
                memset(working_pointer + k % DOUBLE_REGISTER_COUNT, 0, sizeof(double) * (DOUBLE_REGISTER_COUNT - k % DOUBLE_REGISTER_COUNT));
            }
        }
        working_s = _mm512_loadu_pd(working_pointer);
        out[idx] += _mm512_reduce_add_pd(working_s);

        if (k < DOUBLE_REGISTER_COUNT)
        {
            kernel_pointer += k;
            feature_pointer += updated_w;
        }
        else
        {
            if (register_reset_counter > 0)
            {
                feature_pointer += DOUBLE_REGISTER_COUNT;
                kernel_pointer += DOUBLE_REGISTER_COUNT;
            }
            else
            {
                // They reset same direction beginning of next row
                feature_pointer += updated_w - k + 1;
                kernel_pointer += k % DOUBLE_REGISTER_COUNT;
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
