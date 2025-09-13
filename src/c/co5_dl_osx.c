//Copyright (c) 2025, tree-chutes

const unsigned char REGISTER_WIDTH = 4;

unsigned char multiply(unsigned long counter, unsigned long trigger, unsigned long jumpback, double* in, double* kernel, double* out)
{
    double* tmp1;
    __m256d operand1 = _mm256_setzero_pd();
    __m256d operand2 = _mm256_setzero_pd();
    __m256d ret = _mm256_setzero_pd();
    double* tmp0 = in;
    double* tmp2 = out;
    
    for(unsigned long c = 0; c < counter; c++ )
    {
        if (c % trigger == 0)
        {
            tmp1 = kernel;
            if (c != 0)
            {
                tmp0 -= jumpback;
                tmp2 -= jumpback;
            }
        }
        operand1 = _mm256_loadu_pd(tmp0);
        operand2 = _mm256_loadu_pd(tmp1);
        ret = _mm256_mul_pd(operand1, operand2);
        _mm256_storeu_pd(tmp2, ret);
        tmp0 += REGISTER_WIDTH;
        tmp1 += REGISTER_WIDTH;
        tmp2 += REGISTER_WIDTH;
    }
    return 0;
}
