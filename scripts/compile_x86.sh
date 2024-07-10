icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./baseline_AoS.cpp -o baseline_AoS_x86
icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./baseline_SoA.cpp -o baseline_SoA_x86

icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./1_invariant.cpp -o 1_invariant_x86

icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./2_parallel.cpp -o 2_parallel_x86
icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./2_parallel_numa.cpp -o 2_parallel_numa_x86

icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./3_vectorization_pragma_simd.cpp -o 3_vectorization_pragma_simd_x86

icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./4_math_novec.cpp -o 4_math_novec_x86
icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./4_math_x86.cpp -o 4_math_x86

icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./5_math_parallel_x86.cpp -o 5_math_parallel_x86

icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -qopt-report=max ./6_parallel_simd_x86.cpp -o 6_parallel_simd_x86
icpx -O2 -fopenmp -march=icelake-server -fimf-precision=high -fp-model=precise -fimf-precision=low -fimf-domain-exclusion=31 -qopt-report=max ./6_parallel_simd_x86.cpp -o 6_parallel_simd_x86_low_accuracy
