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


void GetOptionPrices(float *pT, float *pS0, float *pK, float *pC, int N)
{
    const float sqrt2 = std::sqrt(2.0f);
    const float rPlusSig2Div2 = r + sig * sig * 0.5f;

    const size_t MAXVL = 4*8;
    int n = N;
    
    size_t vl;
    for (; n > 0; n -= vl) {
        float pTmp[MAXVL];
        vl = vsetvl_e32m4(n);
        
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
        vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
        for (int i = 0; i < vl; i++)
            pTmp[i] = std::log(pTmp[i]);
        vtmp1 = vle_v_f32m4(pTmp, vl);  // load
        
        // tmp1 = tmp1 + (r + sig * sig * 0.5f) * pT[i]
        vtmp1 = vfmacc_vf_f32m4(vtmp1, rPlusSig2Div2, vT, vl);
        
        // tmp2 = std::sqrt(pT[i])
        for (int i = 0; i < vl; i++)
            pTmp[i] = std::sqrt(pT[i]);
        vtmp2 = vle_v_f32m4(pTmp, vl);  // load
        
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
        vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
        for (int i = 0; i < vl; i++)
            pTmp[i] = std::erf(pTmp[i]);
        vtmp1 = vle_v_f32m4(pTmp, vl);  // load
        
        // erf1 = 0.5f + 0.5f * tmp1
        vtmp1 = vfmul_vf_f32m4(vtmp1, 0.5f, vl);
        vres1 = vfadd_vf_f32m4(vtmp1, 0.5f, vl);
        
        // --------------- compute erf2 ---------------
        // float erf2 = 0.5f + 0.5f * std::erf(d2 / std::sqrt(2.0f));
        
        // tmp1 = d2 / std::sqrt(2.0f)
        vtmp1 = vfdiv_vf_f32m4(vres2, sqrt2, vl);
        
        // tmp1 = std::erf(tmp1)
        vse_v_f32m4(pTmp, vtmp1, vl);  // store to use elements
        for (int i = 0; i < vl; i++)
            pTmp[i] = std::erf(pTmp[i]);
        vtmp1 = vle_v_f32m4(pTmp, vl);  // load
        
        // erf2 = 0.5f + 0.5f * tmp1
        vtmp1 = vfmul_vf_f32m4(vtmp1, 0.5f, vl);
        vres2 = vfadd_vf_f32m4(vtmp1, 0.5f, vl);
        
        // --------------- compute C ---------------
        // pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
        
        // tmp2 = (-1.0f) * r * pT[i]
        vtmp2 = vfmul_vf_f32m4(vT, -r, vl);
        
        // tmp2 = std::exp(tmp2), NEED TO OPTIMIZE
        vse_v_f32m4(pTmp, vtmp2, vl);  // store to use elements
        for (int i = 0; i < vl; i++)
            pTmp[i] = std::exp(pTmp[i]);
        vtmp2 = vle_v_f32m4(pTmp, vl);  // load
        
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
