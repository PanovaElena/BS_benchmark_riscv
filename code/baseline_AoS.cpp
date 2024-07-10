
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

struct Option {
    float T = 0.0f;
    float S0 = 0.0f;
    float K = 0.0f;
    float C = 0.0f;
};

Option *pOption;
size_t N, nPackages, packageSize;


void GetOptionPrices(Option *pOption, int N)
{
    for (int i = 0; i < N; i++)
    {
        float d1 = (std::log(pOption[i].S0 / pOption[i].K) + (r + sig * sig * 0.5f) * pOption[i].T) /
            (sig * std::sqrt(pOption[i].T));
        float d2 = d1 - sig * std::sqrt(pOption[i].T);
        float erf1 = 0.5f + std::erf(d1 / std::sqrt(2.0f)) * 0.5f;
        float erf2 = 0.5f + std::erf(d2 / std::sqrt(2.0f)) * 0.5f;
        pOption[i].C = pOption[i].S0 * erf1 - pOption[i].K * std::exp((-1.0f) * r * pOption[i].T) * erf2;
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

    pOption = new Option[N];
    
    for (int i = 0; i < N; i++)
    {
      pOption[i].T = T;
      pOption[i].S0 = S0;
      pOption[i].K = K;
    }
    
    float d1 = (std::log(S0 / K) + (r + sig * sig * 0.5f) * T) / (sig * std::sqrt(T));
    float d2 = d1 - sig * std::sqrt(T);
    float erf1 = 0.5f + std::erf(d1 / std::sqrt(2.0f)) * 0.5f;
    float erf2 = 0.5f + std::erf(d2 / std::sqrt(2.0f)) * 0.5f;
    float expC = S0 * erf1 - K * std::exp((-1.0f) * r * T) * erf2;

    for (size_t package = 0; package < nPackages; package++) {
        size_t start = package * packageSize;
        
        double t0 = omp_get_wtime();
        GetOptionPrices(pOption + start, (int)packageSize);
        double t1 = omp_get_wtime();
        
        bool res = true;
        const float eps = 1e-7;
        float maxDiff = 0.0;
        for (int i = start; i < start + (int)packageSize; i++)
            maxDiff = std::max(maxDiff, std::abs(pOption[i].C - expC));   
        std::cout << "Time: " << t1 - t0 << " s, res = " << (pOption + start)[0].C << " diff " << maxDiff << std::endl;
    }

    delete [] pOption;
    return 0;
}
