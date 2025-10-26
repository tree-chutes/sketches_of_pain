// Copyright (c) 2025, tree-chutes

#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <co5_dl_osx.h>

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

unsigned char softmax_forward_double(unsigned long len, double *input, double *total_log_exp, double *out)
{
    __m512 prediction_s = _mm512_setzero_pd();
    __m512 total_log_exp_s = _mm512_loadu_pd(total_log_exp);

    double *prediction_pointer = input;
    unsigned long count = len / DOUBLE_REGISTER_COUNT + (len % DOUBLE_REGISTER_COUNT != 0);

    for (unsigned long i = 0; i < count; i++)
    {
        prediction_s = _mm512_loadu_pd(prediction_pointer);
        prediction_s = _mm512_div_pd(prediction_s, total_log_exp_s);
        _mm512_storeu_pd(out, prediction_s);
        out += DOUBLE_REGISTER_COUNT;
        prediction_pointer += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}

unsigned char softmax_backward_double(unsigned long len, double *truth, double *prediction, double *out)
{
    __m512 truth_s = _mm512_setzero_pd();
    __m512 prediction_s = _mm512_setzero_pd();

    double *prediction_pointer = prediction;
    double *truth_pointer = truth;

    unsigned long count = len / DOUBLE_REGISTER_COUNT + (len % DOUBLE_REGISTER_COUNT != 0);

    for (unsigned long i = 0; i < count; i++)
    {
        truth_s = _mm512_loadu_pd(truth_pointer);
        prediction_s = _mm512_loadu_pd(prediction_pointer);
        prediction_s = _mm512_sub_pd(truth_s, prediction_s);
        _mm512_storeu_pd(out, prediction_s);
        out += DOUBLE_REGISTER_COUNT;
        prediction_pointer += DOUBLE_REGISTER_COUNT;
        truth_pointer += DOUBLE_REGISTER_COUNT;
    }
    return 0;
}
