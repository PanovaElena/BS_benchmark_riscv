#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <sstream>
#include <string>
#include <iomanip>

const float sig = 0.2f;      // Volatility (0.2 -> 20%)
const float r   = 0.05f;     // Interest rate (0.05 -> 5%)
const float T   = 3.0f;      // Maturity (3 -> 3 years)
const float S0  = 100.0f;    // Initial stock price
const float K   = 100.0f;    // Strike price

float *pT_, *pS0_, *pK_, *pC_;
size_t N_, nPackages, packageSize;

typedef union {
    float f;
    uint32_t i32;
} binary32;

#define FLOAT2INT32(_ri, _rf, x) \
    { binary32 t;        \
      t.f = (x + 12582912.0); \
      _rf = (t.f - 12582912.0);\
      _ri = t.i32 & 0x0000ffff; }  // _ri > 0


float my_exp(float vx){
    
    float Log2 = (float)0x1.62e43p-1;
    float Log2h = (float)0xb.17200p-4;
    float Log2l = (float)0x1.7f7d1cf8p-20;
    float InvLog2 = (float)0x1.715476p0;
    
    // Here should be the tests for exceptional cases
    
    // rounding x/Log2
    
    //t.f = (x*InvLog2 + 12582912.0);
    float vt = vx * InvLog2;
    vt = vt + (float)12582912.0;
    
    //kf = (t.f - 12582912.0);
    float vkf = vt - (float)12582912.0;
    
    // k = t.i32 & 0x0000ffff;
    int vti = *((int*)(&vt));      // cast
    int vk = vti & 0x0000ffff;   // &
    
    // moving x to the small range
    //y = (x - kf*Log2h) - kf*Log2l;      
    float vy = vkf * Log2h;
    vy = vx - vy;
    vkf = vkf * Log2l;
    vy = vy - vkf;
    
    // computing in small range
    //r.f = (float)0x1p0 + y * ((float)0x1p0 + y * ((float)0x1.000000000000bp-1 + 
    //      y * ((float)0x1.5555555555511p-3 + y * ((float)0x1.55555555502a1p-5 + 
    //      y * ((float)0x1.1111111122322p-7 + y * ((float)0x1.6c16c1852b7afp-10 + 
    //      y * ((float)0x1.a01a014761f65p-13 + y * ((float)0x1.a01997c89eb71p-16 + 
    //      y * ((float)0x1.71dee62401315p-19 + y * ((float)0x1.28af3fca213eap-22 + 
    //      y * (float)0x1.ade1569ce2bdfp-26))))))))));
    float vr = vy * (float)0x1.6850e4p-10;
    vr = vr + (float)0x1.123bccp-7;
    vr = vy * vr;
    vr = vr + (float)0x1.555b98p-5;    
    vr = vy * vr;
    vr = vr + (float)0x1.55548ep-3; 
    vr = vy * vr;
    vr = vr + (float)0x1.fffff8p-2; 
    vr = vy * vr;
    vr = vr + (float)0x1p0; 
    vr = vy * vr;
    vr = vr + (float)0x1p0;    
    
    // Exponent update
    //r.i32 += k << 23;
    int vri = *((int*)(&vr)); // cast
    vk = vk << 23;                  // <<
    vri = vri + vk;                // +
    vr = *((float*)(&vri));            // cast
    
    return vr;
}

float my_log(float x)
{
   
        //fast_log abs(rel) : avgError = 2.85911e-06(3.32628e-08), MSE = 4.67298e-06(5.31012e-08), maxError = 1.52588e-05(1.7611e-07)
    float s_log_C0 = -19.645704f;
    float s_log_C1 = 0.767002f;
    float s_log_C2 = 0.3717479f;
    float s_log_C3 = 5.2653985f;
    float s_log_2 = 0.6931472f;
    float s_log_C4 = -(1.0f + s_log_C0) * (1.0f + s_log_C1) / ((1.0f + s_log_C2) * (1.0f + s_log_C3)); //ensures that log(1) == 0

    
// int e = static_cast<int>(ux - 0x3f800000) >> 23;
    int intx = *((int*)(&x));
    int exp = intx - 0x3f800000;
    exp  = exp >> 23;
   // ux |= 0x3f800000;
   // ux &= 0x3fffffff; // 1 <= x < 2  after replacing the exponent field
    intx = intx | 0x3f800000;
    intx = intx & 0x3fffffff;

// x = reinterpret_cast<float&>(ux);
    float y = *((float*)(&intx));

float a = (x + s_log_C0) * (x + s_log_C1);
    //float a = vfadd_vf_f32m4(y, s_log_C0, vl);
    //float tmp = vfadd_vf_f32m4(y, s_log_C1, vl);
    //a = vfmul_vv_f32m4(a, tmp, vl);
float b = (x + s_log_C2) * (x + s_log_C3);

    //float b = vfadd_vf_f32m4(y, s_log_C2, vl);
    //float tmp2 = vfadd_vf_f32m4(y, s_log_C3, vl);
    //b = vfmul_vv_f32m4(b, tmp2, vl);
float c = (float(exp) + s_log_C4);

    //float e = vfcvt_f_x_v_f32m4(exp, vl);
    //float c = vfadd_vf_f32m4(e, s_log_C4, vl);

 float d = a / b;
    //float d = vfdiv_vv_f32m4(a, b, vl);


float result = (c + d)* s_log_2;
    //float c_plus_d = vfadd_vv_f32m4(c, d, vl);
    //float result = vfmul_vf_f32m4(c_plus_d, s_log_2, vl);
    return result;
}

float my_erf(float x)
{
    float a1 = (float)0.254829592;
    float a2 = (float)-0.284496736;
    float a3 = (float)1.421413741;
    float a4 = (float)-1.453152027;
    float a5 = (float)1.061405429;
    float p = (float)0.3275911;
    float float_sign = x > 0? 1.0:-1.0;
    float x_abs= x * float_sign;

    float oper1 = x_abs * p;
    oper1 = oper1 + 1.0f;
    float vect1 = 1.0f;
    float t = vect1 / oper1;

    float oper2 = t * a5, vl;
    oper2 = oper2 + a4;
    oper2 = oper2 * t;
    oper2 = oper2 + a3;
    oper2 = oper2 * t;
    oper2 = oper2 + a2;
    oper2 = oper2 * t;
    oper2 = oper2 + a1;
    oper2 = oper2 * t;

    float minus_x = x * (-1.0f);
    float minus_x_mult_x = x * minus_x;
    float exp_x_x = my_exp(minus_x_mult_x);

    oper2 = oper2 * exp_x_x;
    float y = vect1 - oper2;
    
    
    float y_mult_sign = y * float_sign;
    return y_mult_sign;

}

void GetOptionPrices(float *pT, float *pS0, float *pK, float *pC, int N)
{
    const float sqrt2 = std::sqrt(2.0f);
    const float rPlusSig2Div2 = r + sig * sig * 0.5f;

    const size_t MAXVL = 4*8;
    int n = N;
    
    size_t vl = 1;
    for (; n > 0; n -= vl) {
        
        // --------------- load ---------------
        float vK = pK[0];   
        float vT = pT[0];                     
        float vS0 = pS0[0];
        
        float vtmp1, vtmp2, vres1, vres2;
        
        // --------------- compute d1 ---------------
        // float d1 = (std::log(pS0[i] / pK[i]) + (r + sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
        
        // tmp1 = pS0[i] / pK[i], probably simple division is faster
        vtmp1 = vS0 / vK;
        
        // tmp1 = log(tmp1)
        ////vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::log(pTmp[i]);
        ////vtmp1 = vle_v_f32m4(pTmp, vl);  // load
        vtmp1 = my_log(vtmp1);
        
        // tmp1 = tmp1 + (r + sig * sig * 0.5f) * pT[i]
        vtmp1 = vtmp1 + rPlusSig2Div2*vT;
        
        // tmp2 = std::sqrt(pT[i])
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::sqrt(pT[i]);
        ////vtmp2 = vle_v_f32m4(pTmp, vl);  // load
        vtmp2 = std::sqrt(vT);
        
        // tmp2 = sig * tmp2
        vtmp2 = vtmp2 * sig;
        
        // d1 = tmp1 / tmp2
        vres1 = vtmp1 / vtmp2;
        
        // --------------- compute d2 ---------------
        // float d2 = (std::log(pS0[i] / pK[i]) + (r - sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
        // d1 - d2 = sig*sqrt(T) = tmp2
        // d2 = d1 - tmp2
        
        vres2 = vres1 - vtmp2;
        
        // --------------- compute erf1 ---------------
        // float erf1 = 0.5f + 0.5f * std::erf(d1 / std::sqrt(2.0f));
        
        // tmp1 = d1 / std::sqrt(2.0f)
        vtmp1 = vres1 / sqrt2;
        
        // tmp1 = std::erf(tmp1)
        ////vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::erf(pTmp[i]);
        ////vtmp1 = vle_v_f32m4(pTmp, vl);  // load
        vtmp1 = my_erf(vtmp1);
        
        // erf1 = 0.5f + 0.5f * tmp1
        vtmp1 = vtmp1 * 0.5f;
        vres1 = vtmp1 + 0.5f;
        
        // --------------- compute erf2 ---------------
        // float erf2 = 0.5f + 0.5f * std::erf(d2 / std::sqrt(2.0f));
        
        // tmp1 = d2 / std::sqrt(2.0f)
        vtmp1 = vres2 / sqrt2;
        
        // tmp1 = std::erf(tmp1)
        ////vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::erf(pTmp[i]);
        ////vtmp1 = vle_v_f32m4(pTmp, vl);  // load
        vtmp1 = my_erf(vtmp1);
        
        // erf2 = 0.5f + 0.5f * tmp1
        vtmp1 = vtmp1 * 0.5f;
        vres2 = vtmp1 + 0.5f;
        
        // --------------- compute C ---------------
        // pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
        
        // tmp2 = (-1.0f) * r * pT[i]
        vtmp2 = vT * (-r);
        
        // tmp2 = std::exp(tmp2), NEED TO OPTIMIZE
        ////vse_v_f32m4(pTmp, vtmp2, vl);  // store to use elements
        ////for (int i = 0; i < vl; i++)
        ////    pTmp[i] = std::exp(pTmp[i]);
        ////vtmp2 = vle_v_f32m4(pTmp, vl);  // load
        vtmp2 = my_exp(vtmp2);
              
        // tmp1 = pS0[i] * erf1
        vtmp1 = vS0 * vres1;
        
        // tmp2 = K * tmp2
        vtmp2 = vK * vtmp2;
        
        // tmp2 = tmp1 - tmp2 * erf2, vfnmacc???
        vtmp2 = vtmp2 * vres2;
        vtmp2 = vtmp1 - vtmp2;
        
        // --------------- store ---------------
        pC[0] = vtmp2;
        
        pK += 1;
        pT += 1;
        pS0 += 1;        
        pC += 1;
        
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

    pT_  = new float[4 * N_];
    pK_  = pT_ + N_;
    pS0_ = pT_ + 2 * N_;
    pC_  = pT_ + 3 * N_;
    
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
