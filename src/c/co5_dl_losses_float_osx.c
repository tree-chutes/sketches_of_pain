// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <co5_dl_osx.h>

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

unsigned char softmax_forward_float(unsigned long len, float *input, float *total_log_exp, float *out)
{
    __m512 prediction_s = _mm512_setzero_ps();
    __m512 total_log_exp_s = _mm512_loadu_ps(total_log_exp);

    float *prediction_pointer = input;
    unsigned long count = len / FLOAT_REGISTER_COUNT + (len % FLOAT_REGISTER_COUNT != 0);

    for (unsigned long i = 0; i < count; i++)
    {
        prediction_s = _mm512_loadu_ps(prediction_pointer);
        prediction_s = _mm512_div_ps(prediction_s, total_log_exp_s);
        _mm512_storeu_ps(out, prediction_s);
        out += FLOAT_REGISTER_COUNT;
        prediction_pointer += FLOAT_REGISTER_COUNT;
    }
    return 0;
}

unsigned char softmax_backward_float(unsigned long len, float *truth, float *prediction, float *out)
{
    __m512 truth_s = _mm512_setzero_ps();
    __m512 prediction_s = _mm512_setzero_ps();

    float *prediction_pointer = prediction;
    float *truth_pointer = truth;

    unsigned long count = len / FLOAT_REGISTER_COUNT + (len % FLOAT_REGISTER_COUNT != 0);

    for (unsigned long i = 0; i < count; i++)
    {
        truth_s = _mm512_loadu_ps(truth_pointer);
        prediction_s = _mm512_loadu_ps(prediction_pointer);
        prediction_s = _mm512_sub_ps(truth_s, prediction_s);
        _mm512_storeu_ps(out, prediction_s);
        out += FLOAT_REGISTER_COUNT;
        prediction_pointer += FLOAT_REGISTER_COUNT;
        truth_pointer += FLOAT_REGISTER_COUNT;
    }
    return 0;
}
