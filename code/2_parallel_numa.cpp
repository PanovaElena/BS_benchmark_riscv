
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

float *pT, *pS0, *pK, *pC;
size_t N, nPackages, packageSize;


void GetOptionPrices(float *pT, float *pS0, float *pK, float *pC, int N)
{
    const float sqrt2 = std::sqrt(2.0f);
    const float rPlusSig2Div2 = r + sig * sig * 0.5f;
    
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        float sigSqrtT = sig * std::sqrt(pT[i]);
        float d1 = (std::log(pS0[i] / pK[i]) + rPlusSig2Div2 * pT[i]) / sigSqrtT;
        float d2 = d1 - sigSqrtT;
        float erf1 = 0.5f + std::erf(d1 / sqrt2) * 0.5f;
        float erf2 = 0.5f + std::erf(d2 / sqrt2) * 0.5f;
        pC[i] = pS0[i] * erf1 - pK[i] * std::exp((-1.0f) * r * pT[i]) * erf2;
    }
}


int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "WRONG ARGS" << std::endl;
        return 0;
    }

    std::stringstream sstr(argv[1]);
    sstr >> N;
    nPackages = atoi(argv[2]);
    packageSize = N / nPackages;

    pT  = new float[4 * N];
    pK  = pT + N;
    pS0 = pT + 2 * N;
    pC  = pT + 3 * N;
    
    float d1 = (std::log(S0 / K) + (r + sig * sig * 0.5f) * T) / (sig * std::sqrt(T));
    float d2 = d1 - sig * std::sqrt(T);
    float erf1 = 0.5f + std::erf(d1 / std::sqrt(2.0f)) * 0.5f;
    float erf2 = 0.5f + std::erf(d2 / std::sqrt(2.0f)) * 0.5f;
    float expC = S0 * erf1 - K * std::exp((-1.0f) * r * T) * erf2;

    for (size_t package = 0; package < nPackages; package++) {
        size_t start = package * packageSize;
        
#pragma omp parallel for
        for (int i = start; i < start + packageSize; i++)
        {
            pT[i] = T;
            pS0[i] = S0;
            pK[i] = K;
            pC[i] = 0.0f;
        }
        
        double t0 = omp_get_wtime();
        GetOptionPrices(pT + start, pK + start, pS0 + start, pC + start, (int)packageSize);
        double t1 = omp_get_wtime();
        
        bool res = true;
        const float eps = 1e-7;
        float maxDiff = 0.0;
        for (int i = start; i < start + (int)packageSize; i++)
            maxDiff = std::max(maxDiff, std::abs(pC[i] - expC));   
        std::cout << "Time: " << t1 - t0 << " s, res = " << (pC + start)[0] << " diff " << maxDiff << std::endl;
    }

    delete [] pT;
    return 0;
}
