#include <immintrin.h>
#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <sstream>
#include <string>

const float sig = 0.2f;      // Volatility (0.2 -> 20%)
const float r   = 0.05f;     // Interest rate (0.05 -> 5%)
const float T   = 3.0f;      // Maturity (3 -> 3 years)
const float S0  = 100.0f;    // Initial stock price
const float K   = 100.0f;    // Strike price

float *pT_, *pS0_, *pK_, *pC_;
size_t N_, nPackages, packageSize;

__m512 my_exp(__m512 vx){
    
    float Log2 = (float)0x1.62e43p-1;
    float Log2h = (float)0xb.17200p-4;
    float Log2l = (float)0x1.7f7d1cf8p-20;
    float InvLog2 = (float)0x1.715476p0;
    
    // Here should be the tests for exceptional cases
    
    // rounding x/Log2
    
    //t.f = (x*InvLog2 + 12582912.0);
    __m512 tmpConst = _mm512_set1_ps(InvLog2);
    __m512 vt = _mm512_mul_ps(vx, tmpConst);
    tmpConst = _mm512_set1_ps(12582912.0f);
    vt = _mm512_add_ps(vt, tmpConst);
    
    //kf = (t.f - 12582912.0);
    __m512 vkf = _mm512_sub_ps(vt, tmpConst);
    
    // k = t.i32 & 0x0000ffff;
    __m512i vti = _mm512_castps_si512(vt);      // cast
    __m512i tmpConsti = _mm512_set1_epi32(0x0000ffff);
    __m512i vk = _mm512_and_ps(vti, tmpConsti);   // &
    
    // moving x to the small range
    //y = (x - kf*Log2h) - kf*Log2l;
    tmpConst = _mm512_set1_ps(Log2h);    
    __m512 vy = _mm512_mul_ps(vkf, tmpConst);
    vy = _mm512_sub_ps(vx, vy);
    tmpConst = _mm512_set1_ps(Log2l); 
    vkf = _mm512_mul_ps(vkf, tmpConst);
    vy = _mm512_sub_ps(vy, vkf);
    
    // computing in small range
    //r.f = (float)0x1p0 + y * ((float)0x1p0 + y * ((float)0x1.000000000000bp-1 + 
    //      y * ((float)0x1.5555555555511p-3 + y * ((float)0x1.55555555502a1p-5 + 
    //      y * ((float)0x1.1111111122322p-7 + y * ((float)0x1.6c16c1852b7afp-10 + 
    //      y * ((float)0x1.a01a014761f65p-13 + y * ((float)0x1.a01997c89eb71p-16 + 
    //      y * ((float)0x1.71dee62401315p-19 + y * ((float)0x1.28af3fca213eap-22 + 
    //      y * (float)0x1.ade1569ce2bdfp-26))))))))));
    tmpConst = _mm512_set1_ps((float)0x1.6850e4p-10);
    __m512 vr = _mm512_mul_ps(vy, tmpConst);
    tmpConst = _mm512_set1_ps((float)0x1.123bccp-7);
    vr = _mm512_add_ps(vr, tmpConst);
    vr = _mm512_mul_ps(vy, vr);
    tmpConst = _mm512_set1_ps((float)0x1.555b98p-5);
    vr = _mm512_add_ps(vr, tmpConst);    
    vr = _mm512_mul_ps(vy, vr);
    tmpConst = _mm512_set1_ps((float)0x1.55548ep-3);
    vr = _mm512_add_ps(vr, tmpConst); 
    vr = _mm512_mul_ps(vy, vr);
    tmpConst = _mm512_set1_ps((float)0x1.fffff8p-2);
    vr = _mm512_add_ps(vr, tmpConst); 
    vr = _mm512_mul_ps(vy, vr);
    tmpConst = _mm512_set1_ps(0x1p0);
    vr = _mm512_add_ps(vr, tmpConst); 
    vr = _mm512_mul_ps(vy, vr);
    vr = _mm512_add_ps(vr, tmpConst);    
    
    // Exponent update
    //r.i32 += k << 23;
    __m512i vri = _mm512_castps_si512(vr); // cast
    tmpConsti = _mm512_set1_epi32(23);
    vk = _mm512_sllv_epi32(vk, tmpConsti);                  // <<
    vri = _mm512_add_epi32(vri, vk);                // +
    vr = _mm512_castsi512_ps(vri);            // cast
    
    return vr;
}

__m512 my_log(__m512 x)
{
   
        //fast_log abs(rel) : avgError = 2.85911e-06(3.32628e-08), MSE = 4.67298e-06(5.31012e-08), maxError = 1.52588e-05(1.7611e-07)
    float s_log_C0 = -19.645704f;
    float s_log_C1 = 0.767002f;
    float s_log_C2 = 0.3717479f;
    float s_log_C3 = 5.2653985f;
    float s_log_2 = 0.6931472f;
    float s_log_C4 = -(1.0f + s_log_C0) * (1.0f + s_log_C1) / ((1.0f + s_log_C2) * (1.0f + s_log_C3)); //ensures that log(1) == 0

    
// int e = static_cast<int>(ux - 0x3f800000) >> 23;
    __m512i intx = _mm512_castps_si512(x);
    __m512i tmpConsti = _mm512_set1_epi32(0x3f800000);
    __m512i exp = _mm512_sub_epi32(intx, tmpConsti);
    __m512i tmpConstiShift = _mm512_set1_epi32(23);
    exp  = _mm512_srav_epi32(exp, tmpConstiShift);
   // ux |= 0x3f800000;
   // ux &= 0x3fffffff; // 1 <= x < 2  after replacing the exponent field
    intx = _mm512_or_epi32(intx, tmpConsti);
    tmpConsti = _mm512_set1_epi32(0x3fffffff);
    intx = _mm512_and_epi32(intx, tmpConsti);

// x = reinterpret_cast<float&>(ux);
    __m512 y = _mm512_castsi512_ps(intx);
    
//float a = (x + s_log_C0) * (x + s_log_C1);
    __m512 tmpConst = _mm512_set1_ps(s_log_C0);
    __m512 a = _mm512_add_ps(y, tmpConst);
    tmpConst = _mm512_set1_ps(s_log_C1);
    __m512 tmp = _mm512_add_ps(y, tmpConst);
    a = _mm512_mul_ps(a, tmp);
// float b = (x + s_log_C2) * (x + s_log_C3);

    tmpConst = _mm512_set1_ps(s_log_C2);
    __m512 b = _mm512_add_ps(y, tmpConst);
    tmpConst = _mm512_set1_ps(s_log_C3);
    __m512 tmp2 = _mm512_add_ps(y, tmpConst);
    b = _mm512_mul_ps(b, tmp2);
//float c = (float(e) + s_log_C4);

    __m512 e = _mm512_cvtepi32_ps(exp);//vfcvt_f_x_v_f32m4(exp);
    tmpConst = _mm512_set1_ps(s_log_C4);
    __m512 c = _mm512_add_ps(e, tmpConst);

// float d = a / b;
    __m512 d = _mm512_div_ps(a, b);


//(c + d)* s_log_2;
    __m512 c_plus_d = _mm512_add_ps(c, d);
    tmpConst = _mm512_set1_ps(s_log_2);
    __m512 result = _mm512_mul_ps(c_plus_d, tmpConst);
    return result;
}

__m512 my_erf(__m512 x)
{
    float a1 = (float)0.254829592;
    float a2 = (float)-0.284496736;
    float a3 = (float)1.421413741;
    float a4 = (float)-1.453152027;
    float a5 = (float)1.061405429;
    float p = (float)0.3275911;
    __m512i sign;
    __m512i floatcast = _mm512_castps_si512(x);
    __m512i tmpConstiShift = _mm512_set1_epi32(31);
    sign  = _mm512_srav_epi32(floatcast, tmpConstiShift);
    __m512i tmpConsti = _mm512_set1_epi32(1);
    sign = _mm512_or_epi32(sign, tmpConsti);
    __m512 float_sign = _mm512_cvtepi32_ps(sign);

    __m512 x_abs= _mm512_mul_ps(float_sign, x);

    __m512 tmpConst = _mm512_set1_ps(p);
    __m512 oper1 = _mm512_mul_ps(x_abs, tmpConst);
    __m512 tmpConst1 = _mm512_set1_ps(1.0f);
    oper1 = _mm512_add_ps(oper1, tmpConst1);
    __m512 t = _mm512_div_ps(tmpConst1, oper1);
    
    tmpConst = _mm512_set1_ps(a5);
    __m512 oper2 = _mm512_mul_ps(t, tmpConst);
    tmpConst = _mm512_set1_ps(a4);
    oper2 = _mm512_add_ps(oper2, tmpConst);
    oper2 = _mm512_mul_ps(oper2, t);
    tmpConst = _mm512_set1_ps(a3);
    oper2 = _mm512_add_ps(oper2, tmpConst);
    oper2 = _mm512_mul_ps(oper2, t);
    tmpConst = _mm512_set1_ps(a2);
    oper2 = _mm512_add_ps(oper2, tmpConst);
    oper2 = _mm512_mul_ps(oper2, t);
    tmpConst = _mm512_set1_ps(a1);
    oper2 = _mm512_add_ps(oper2, tmpConst);
    oper2 = _mm512_mul_ps(oper2, t);

    tmpConst = _mm512_set1_ps(-1.0f);
    __m512 minus_x = _mm512_mul_ps(x, tmpConst);
    __m512 minus_x_mult_x = _mm512_mul_ps(x, minus_x);
    __m512 exp_x_x = my_exp(minus_x_mult_x);

    oper2 = _mm512_mul_ps(oper2, exp_x_x);
    __m512 y = _mm512_sub_ps(tmpConst1, oper2);
    
    
    __m512 y_mult_sign = _mm512_mul_ps(y, float_sign);
    return y_mult_sign;

}

void GetOptionPrices(float *pT, float *pS0, float *pK, float *pC, int N)
{
    const float sqrt2 = std::sqrt(2.0f);
    const float rPlusSig2Div2 = r + sig * sig * 0.5f;

    const size_t MAXVL = 16;
    int n = N;
    
    size_t vl = MAXVL;
    for (; n > 0; n -= vl) {
        float pTmp[MAXVL];
        
        // --------------- load ---------------
        __m512 vK = _mm512_load_pd(pK);   
        __m512 vT = _mm512_load_pd(pT);                     
        __m512 vS0 = _mm512_load_pd(pS0);
        
        __m512 vtmp1, vtmp2, vres1, vres2;
        
        // --------------- compute d1 ---------------
        // float d1 = (std::log(pS0[i] / pK[i]) + (r + sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
        
        // tmp1 = pS0[i] / pK[i], probably simple division is faster
        vtmp1 = _mm512_div_ps(vS0, vK);
        
        // tmp1 = log(tmp1)
        ////vse_v_f32m4(pTmp, vtmp1);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::log(pTmp[i]);
        ////vtmp1 = vle_v_f32m4(pTmp);  // load
        vtmp1 = my_log(vtmp1);
        
        // tmp1 = tmp1 + (r + sig * sig * 0.5f) * pT[i]
        __m512 tmpConst = _mm512_set1_ps(rPlusSig2Div2);
        __m512 tmp = _mm512_mul_ps(vT, tmpConst);
        vtmp1 = _mm512_add_ps(vtmp1, tmp);
        
        // tmp2 = std::sqrt(pT[i])
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::sqrt(pT[i]);
        ////vtmp2 = vle_v_f32m4(pTmp);  // load
        vtmp2 = _mm512_sqrt_ps(vT);
        
        // tmp2 = sig * tmp2
        tmpConst = _mm512_set1_ps(sig);
        vtmp2 = _mm512_mul_ps(vtmp2, tmpConst);
        
        // d1 = tmp1 / tmp2
        vres1 = _mm512_div_ps(vtmp1, vtmp2);
        
        // --------------- compute d2 ---------------
        // float d2 = (std::log(pS0[i] / pK[i]) + (r - sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
        // d1 - d2 = sig*sqrt(T) = tmp2
        // d2 = d1 - tmp2
        
        vres2 = _mm512_sub_ps(vres1, vtmp2);
        
        // --------------- compute erf1 ---------------
        // float erf1 = 0.5f + 0.5f * std::erf(d1 / std::sqrt(2.0f));
        
        // tmp1 = d1 / std::sqrt(2.0f)
        __m512 tmpConstSqrt2 = _mm512_set1_ps(sqrt2);
        vtmp1 = _mm512_div_ps(vres1, tmpConstSqrt2);
        
        // tmp1 = std::erf(tmp1)
        ////vse_v_f32m4(pTmp, vtmp1);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::erf(pTmp[i]);
        ////vtmp1 = vle_v_f32m4(pTmp);  // load
        vtmp1 = my_erf(vtmp1);
        
        // erf1 = 0.5f + 0.5f * tmp1
        tmpConst = _mm512_set1_ps(0.5f);
        vtmp1 = _mm512_mul_ps(vtmp1, tmpConst);
        vres1 = _mm512_add_ps(vtmp1, tmpConst);
        
        // --------------- compute erf2 ---------------
        // float erf2 = 0.5f + 0.5f * std::erf(d2 / std::sqrt(2.0f));
        
        // tmp1 = d2 / std::sqrt(2.0f)
        vtmp1 = _mm512_div_ps(vres2, tmpConstSqrt2);
        
        // tmp1 = std::erf(tmp1)
        ////vse_v_f32m4(pTmp, vtmp1);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::erf(pTmp[i]);
        ////vtmp1 = vle_v_f32m4(pTmp);  // load
        vtmp1 = my_erf(vtmp1);
        
        // erf2 = 0.5f + 0.5f * tmp1
        vtmp1 = _mm512_mul_ps(vtmp1, tmpConst);
        vres2 = _mm512_add_ps(vtmp1, tmpConst);
        
        // --------------- compute C ---------------
        // pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
        
        // tmp2 = (-1.0f) * r * pT[i]
        tmpConst = _mm512_set1_ps(-r);
        vtmp2 = _mm512_mul_ps(vT, tmpConst);
        
        // tmp2 = std::exp(tmp2), NEED TO OPTIMIZE
        ////vse_v_f32m4(pTmp, vtmp2);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::exp(pTmp[i]);
        ////vtmp2 = vle_v_f32m4(pTmp);  // load
        vtmp2 = my_exp(vtmp2);
        
        // tmp2 = tmp2 * erf2
        vtmp2 = _mm512_mul_ps(vtmp2, vres2);
              
        // tmp1 = pS0[i] * erf1
        vtmp1 = _mm512_mul_ps(vS0, vres1);
        
        // tmp2 = tmp1 - tmp2 * pK[i], vfnmacc???
        vtmp2 = _mm512_mul_ps(vtmp2, vK);
        vtmp2 = _mm512_sub_ps(vtmp1, vtmp2);
        
        // --------------- store ---------------
        _mm512_store_ps(pC, vtmp2);
        
        pK += vl;
        pT += vl;
        pS0 += vl;        
        pC += vl;
        
        //float d1 = (std::log(pS0[i] / pK[i]) + (r + sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
        //float d2 = (std::log(pS0[i] / pK[i]) + (r - sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
        //float erf1 = 0.5f + 0.5f * std::erf(d1 / std::sqrt(2.0f));
        //float erf2 = 0.5f + 0.5f * std::erf(d2 / std::sqrt(2.0f));
        //pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
    }
}


int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "WRONG ARGS" << std::endl;
        return 0;
    }

    std::stringstream sstr(argv[1]);
    sstr >> N_;
    nPackages = atoi(argv[2]);
    packageSize = N_ / nPackages;

    pT_  = new float[4 * (N_ + 16)];
    pK_  = pT_ + (N_ + 16);
    pS0_ = pT_ + 2 * (N_ + 16);
    pC_  = pT_ + 3 * (N_ + 16);
    
    float d1 = (std::log(S0 / K) + (r + sig * sig * 0.5f) * T) / (sig * std::sqrt(T));
    float d2 = d1 - sig * std::sqrt(T);
    float erf1 = 0.5f + std::erf(d1 / std::sqrt(2.0f)) * 0.5f;
    float erf2 = 0.5f + std::erf(d2 / std::sqrt(2.0f)) * 0.5f;
    float expC = S0 * erf1 - K * std::exp((-1.0f) * r * T) * erf2;

    for (size_t package = 0; package < nPackages; package++) {
        size_t start = package * packageSize;
        
        for (int i = start; i < start + packageSize; i++)
        {
            pT_[i] = T;
            pS0_[i] = S0;
            pK_[i] = K;
            pC_[i] = 0.0;
        }
        
        double t0 = omp_get_wtime();
        GetOptionPrices(pT_ + start, pK_ + start, pS0_ + start, pC_ + start, (int)packageSize);
        double t1 = omp_get_wtime();
        
        bool res = true;
        const float eps = 1e-7;
        float maxDiff = 0.0;
        for (int i = start; i < start + (int)packageSize; i++)
            maxDiff = std::max(maxDiff, std::abs(pC_[i] - expC));   
        std::cout << "Time: " << t1 - t0 << " s, res = " << (pC_ + start)[0] << " diff " << maxDiff << std::endl;
    }

    delete [] pT_;
    return 0;
}
