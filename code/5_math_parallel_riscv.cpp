#include <riscv-vector.h>
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

vfloat32m4_t my_exp(vfloat32m4_t vx, size_t vl){
    
    float Log2 = (float)0x1.62e43p-1;
    float Log2h = (float)0xb.17200p-4;
    float Log2l = (float)0x1.7f7d1cf8p-20;
    float InvLog2 = (float)0x1.715476p0;
    
    // Here should be the tests for exceptional cases
    
    // rounding x/Log2
    
    //t.f = (x*InvLog2 + 12582912.0);
    vfloat32m4_t vt = vfmul_vf_f32m4(vx, InvLog2, vl);
    vt = vfadd_vf_f32m4(vt, (float)12582912.0, vl);
    
    //kf = (t.f - 12582912.0);
    vfloat32m4_t vkf = vfsub_vf_f32m4(vt, (float)12582912.0, vl);
    
    // k = t.i32 & 0x0000ffff;
    vint32m4_t vti = vreinterpret_v_f32m4_i32m4(vt);      // cast
    vint32m4_t vk = vand_vx_i32m4(vti, 0x0000ffff, vl);   // &
    
    // moving x to the small range
    //y = (x - kf*Log2h) - kf*Log2l;      
    vfloat32m4_t vy = vfmul_vf_f32m4(vkf, Log2h, vl);
    vy = vfsub_vv_f32m4(vx, vy, vl);
    vkf = vfmul_vf_f32m4(vkf, Log2l, vl);
    vy = vfsub_vv_f32m4(vy, vkf, vl);
    
    // computing in small range
    //r.f = (float)0x1p0 + y * ((float)0x1p0 + y * ((float)0x1.000000000000bp-1 + 
    //      y * ((float)0x1.5555555555511p-3 + y * ((float)0x1.55555555502a1p-5 + 
    //      y * ((float)0x1.1111111122322p-7 + y * ((float)0x1.6c16c1852b7afp-10 + 
    //      y * ((float)0x1.a01a014761f65p-13 + y * ((float)0x1.a01997c89eb71p-16 + 
    //      y * ((float)0x1.71dee62401315p-19 + y * ((float)0x1.28af3fca213eap-22 + 
    //      y * (float)0x1.ade1569ce2bdfp-26))))))))));
    vfloat32m4_t vr = vfmul_vf_f32m4(vy, (float)0x1.6850e4p-10, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.123bccp-7, vl);
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.555b98p-5, vl);    
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.55548ep-3, vl); 
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1.fffff8p-2, vl); 
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1p0, vl); 
    vr = vfmul_vv_f32m4(vy, vr, vl);
    vr = vfadd_vf_f32m4(vr, (float)0x1p0, vl);    
    
    // Exponent update
    //r.i32 += k << 23;
    vint32m4_t vri = vreinterpret_v_f32m4_i32m4(vr); // cast
    vk = vsll_vx_i32m4(vk, 23, vl);                  // <<
    vri = vadd_vv_i32m4(vri, vk, vl);                // +
    vr = vreinterpret_v_i32m4_f32m4(vri);            // cast
    
    return vr;
}

vfloat32m4_t my_log(vfloat32m4_t x, int vl)
{
   
    //fast_log abs(rel) : avgError = 2.85911e-06(3.32628e-08), MSE = 4.67298e-06(5.31012e-08), maxError = 1.52588e-05(1.7611e-07)
    float s_log_C0 = -19.645704f;
    float s_log_C1 = 0.767002f;
    float s_log_C2 = 0.3717479f;
    float s_log_C3 = 5.2653985f;
    float s_log_2 = 0.6931472f;
    float s_log_C4 = -(1.0f + s_log_C0) * (1.0f + s_log_C1) / ((1.0f + s_log_C2) * (1.0f + s_log_C3)); //ensures that log(1) == 0

    
    // int e = static_cast<int>(ux - 0x3f800000) >> 23;
    vint32m4_t intx = vreinterpret_v_f32m4_i32m4(x);
    vint32m4_t exp = vsub_vx_i32m4(intx, 0x3f800000, vl);
    exp  = vsra_vx_i32m4(exp, 23, vl);
    // ux |= 0x3f800000;
    // ux &= 0x3fffffff; // 1 <= x < 2  after replacing the exponent field
    intx = vor_vx_i32m4(intx, 0x3f800000, vl);
    intx = vand_vx_i32m4(intx, 0x3fffffff, vl);

    // x = reinterpret_cast<float&>(ux);
    vfloat32m4_t y = vreinterpret_v_i32m4_f32m4(intx);

    //float a = (x + s_log_C0) * (x + s_log_C1);
    vfloat32m4_t a = vfadd_vf_f32m4(y, s_log_C0, vl);
    vfloat32m4_t tmp = vfadd_vf_f32m4(y, s_log_C1, vl);
    a = vfmul_vv_f32m4(a, tmp, vl);
    // float b = (x + s_log_C2) * (x + s_log_C3);

    vfloat32m4_t b = vfadd_vf_f32m4(y, s_log_C2, vl);
    vfloat32m4_t tmp2 = vfadd_vf_f32m4(y, s_log_C3, vl);
    b = vfmul_vv_f32m4(b, tmp2, vl);
    //float c = (float(e) + s_log_C4);

    vfloat32m4_t e = vfcvt_f_x_v_f32m4(exp, vl);
    vfloat32m4_t c = vfadd_vf_f32m4(e, s_log_C4, vl);

    // float d = a / b;
    vfloat32m4_t d = vfdiv_vv_f32m4(a, b, vl);

    //(c + d)* s_log_2;
    vfloat32m4_t c_plus_d = vfadd_vv_f32m4(c, d, vl);
    vfloat32m4_t result = vfmul_vf_f32m4(c_plus_d, s_log_2, vl);
    return result;
}

vfloat32m4_t my_erf(vfloat32m4_t x, int vl)
{
    float a1 = (float)0.254829592;
    float a2 = (float)-0.284496736;
    float a3 = (float)1.421413741;
    float a4 = (float)-1.453152027;
    float a5 = (float)1.061405429;
    float p = (float)0.3275911;
    vint32m4_t sign;
    vint32m4_t floatcast = vreinterpret_v_f32m4_i32m4(x);
    sign = vsra_vx_i32m4(floatcast, 31, vl);
    sign = vor_vx_i32m4(sign, 1, vl);
    vfloat32m4_t float_sign =  vfcvt_f_x_v_f32m4(sign,  vl);

    vfloat32m4_t x_abs= vfmul_vv_f32m4(float_sign, x, vl);

    vfloat32m4_t oper1 = vfmul_vf_f32m4(x_abs, p, vl);
    oper1 = vfadd_vf_f32m4(oper1, 1.0f, vl);
    vfloat32m4_t vect1 = vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t t = vfdiv_vv_f32m4(vect1, oper1, vl);

    vfloat32m4_t oper2 = vfmul_vf_f32m4(t, a5, vl);
    oper2 = vfadd_vf_f32m4(oper2, a4, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);
    oper2 = vfadd_vf_f32m4(oper2, a3, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);
    oper2 = vfadd_vf_f32m4(oper2, a2, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);
    oper2 = vfadd_vf_f32m4(oper2, a1, vl);
    oper2 = vfmul_vv_f32m4(oper2, t, vl);

    vfloat32m4_t minus_x = vfmul_vf_f32m4(x, -1.0f, vl);
    vfloat32m4_t minus_x_mult_x = vfmul_vv_f32m4(x, minus_x, vl);
    vfloat32m4_t exp_x_x = my_exp(minus_x_mult_x, vl);

    oper2 = vfmul_vv_f32m4(oper2, exp_x_x, vl);
    vfloat32m4_t y = vfsub_vv_f32m4(vect1, oper2, vl);
    
    
    vfloat32m4_t y_mult_sign = vfmul_vv_f32m4(y, float_sign, vl);
    return y_mult_sign;

}

void GetOptionPrices(float *_pT, float *_pS0, float *_pK, float *_pC, int N)
{
    const float sqrt2 = std::sqrt(2.0f);
    const float rPlusSig2Div2 = r + sig * sig * 0.5f;

    const size_t MAXVL = 4*8;
    int n = N;
    
    int block_size = 0;
    if (n < omp_get_max_threads()) block_size = n;
	else block_size = n / omp_get_max_threads();
    
    int block_number = 0;
    int extra_elements = 0;
	if (n % block_size == 0)
        block_number = n / block_size;
	else {
        block_number = n / block_size;
        extra_elements = n - block_size * block_number;
	}
    
    #pragma omp parallel for
    for (int block_index = 0; block_index < block_number; block_index++) { 
        int block_start = block_index * block_size;
        int block_end = block_start + block_size;
        int last_block_size = 0;
        if (block_number - block_index == 1 && extra_elements != 0)
            block_end = n;
        
        float* pC = _pC + block_start, *pK = _pK + block_start, *pS0 = _pS0 + block_start, *pT = _pT + block_start;
        
        size_t vl;
        while (block_start < block_end) {
            vl = vsetvl_e32m4(block_end - block_start);
            
            // --------------- load ---------------
            vfloat32m4_t vK = vle_v_f32m4(pK, vl);   
            vfloat32m4_t vT = vle_v_f32m4(pT, vl);                     
            vfloat32m4_t vS0 = vle_v_f32m4(pS0, vl);
            
            vfloat32m4_t vtmp1, vtmp2, vres1, vres2;
            
            // --------------- compute d1 ---------------
            // float d1 = (std::log(pS0[i] / pK[i]) + (r + sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
            
            // tmp1 = pS0[i] / pK[i], probably simple division is faster
            vtmp1 = vfdiv_vv_f32m4(vS0, vK, vl);
            
            // tmp1 = log(tmp1)
            ////vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
            ////for (int i = 0; i < vl; i++)
            ////    pTmp[i] = std::log(pTmp[i]);
            ////vtmp1 = vle_v_f32m4(pTmp, vl);  // load
            vtmp1 = my_log(vtmp1, vl);
            
            // tmp1 = tmp1 + (r + sig * sig * 0.5f) * pT[i]
            vtmp1 = vfmacc_vf_f32m4(vtmp1, rPlusSig2Div2, vT, vl);
            
            // tmp2 = std::sqrt(pT[i])
            ////for (int i = 0; i < vl; i++)
            ////    pTmp[i] = std::sqrt(pT[i]);
            ////vtmp2 = vle_v_f32m4(pTmp, vl);  // load
            vtmp2 = vfsqrt_v_f32m4(vT, vl);
            
            // tmp2 = sig * tmp2
            vtmp2 = vfmul_vf_f32m4(vtmp2, sig, vl);
            
            // d1 = tmp1 / tmp2
            vres1 = vfdiv_vv_f32m4(vtmp1, vtmp2, vl);
            
            // --------------- compute d2 ---------------
            // float d2 = (std::log(pS0[i] / pK[i]) + (r - sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
            // d1 - d2 = sig*sqrt(T) = tmp2
            // d2 = d1 - tmp2
            
            vres2 = vfsub_vv_f32m4(vres1, vtmp2, vl);
            
            // --------------- compute erf1 ---------------
            // float erf1 = 0.5f + 0.5f * std::erf(d1 / std::sqrt(2.0f));
            
            // tmp1 = d1 / std::sqrt(2.0f)
            vtmp1 = vfdiv_vf_f32m4(vres1, sqrt2, vl);
            
            // tmp1 = std::erf(tmp1)
            ////vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
            ////for (int i = 0; i < vl; i++)
            ////    pTmp[i] = std::erf(pTmp[i]);
            ////vtmp1 = vle_v_f32m4(pTmp, vl);  // load
            vtmp1 = my_erf(vtmp1, vl);
            
            // erf1 = 0.5f + 0.5f * tmp1
            vtmp1 = vfmul_vf_f32m4(vtmp1, 0.5f, vl);
            vres1 = vfadd_vf_f32m4(vtmp1, 0.5f, vl);
            
            // --------------- compute erf2 ---------------
            // float erf2 = 0.5f + 0.5f * std::erf(d2 / std::sqrt(2.0f));
            
            // tmp1 = d2 / std::sqrt(2.0f)
            vtmp1 = vfdiv_vf_f32m4(vres2, sqrt2, vl);
            
            // tmp1 = std::erf(tmp1)
            ////vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
            ////for (int i = 0; i < vl; i++)
            ////    pTmp[i] = std::erf(pTmp[i]);
            ////vtmp1 = vle_v_f32m4(pTmp, vl);  // load
            vtmp1 = my_erf(vtmp1, vl);
            
            // erf2 = 0.5f + 0.5f * tmp1
            vtmp1 = vfmul_vf_f32m4(vtmp1, 0.5f, vl);
            vres2 = vfadd_vf_f32m4(vtmp1, 0.5f, vl);
            
            // --------------- compute C ---------------
            // pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
            
            // tmp2 = (-1.0f) * r * pT[i]
            vtmp2 = vfmul_vf_f32m4(vT, -r, vl);
            
            // tmp2 = std::exp(tmp2), NEED TO OPTIMIZE
            ////vse_v_f32m4(pTmp, vtmp2, vl);  // store to use elements
            ////for (int i = 0; i < vl; i++)
            ////    pTmp[i] = std::exp(pTmp[i]);
            ////vtmp2 = vle_v_f32m4(pTmp, vl);  // load
            vtmp2 = my_exp(vtmp2, vl);
            
            // tmp2 = tmp2 * erf2
            vtmp2 = vfmul_vv_f32m4(vtmp2, vres2, vl);
                
            // tmp1 = pS0[i] * erf1
            vtmp1 = vfmul_vv_f32m4(vS0, vres1, vl);
            
            // tmp2 = tmp1 - tmp2 * pK[i], vfnmacc???
            vtmp2 = vfmul_vv_f32m4(vtmp2, vK, vl);
            vtmp2 = vfsub_vv_f32m4(vtmp1, vtmp2, vl);
            
            // --------------- store ---------------
            vse_v_f32m4(pC, vtmp2, vl);
            
            pK += vl;
            pT += vl;
            pS0 += vl;        
            pC += vl;
            block_start += vl;
            
            //float d1 = (std::log(pS0[i] / pK[i]) + (r + sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
            //float d2 = (std::log(pS0[i] / pK[i]) + (r - sig * sig * 0.5f) * pT[i]) / (sig * std::sqrt(pT[i]));
            //float erf1 = 0.5f + 0.5f * std::erf(d1 / std::sqrt(2.0f));
            //float erf2 = 0.5f + 0.5f * std::erf(d2 / std::sqrt(2.0f));
            //pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
        }
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
