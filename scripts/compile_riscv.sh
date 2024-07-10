riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./baseline_AoS.cpp -o baseline_AoS_riscv
riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./baseline_SoA.cpp -o baseline_SoA_riscv

riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./1_invariant.cpp -o 1_invariant_riscv

riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./2_parallel.cpp -o 2_parallel_riscv
riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./2_parallel_numa.cpp -o 2_parallel_numa_riscv

riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./3_vectorization_pragma_simd.cpp -o 3_vectorization_pragma_simd_riscv
riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./3_vectorization_intrinsic_riscv.cpp -o 3_vectorization_intrinsic_riscv

riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./4_math_novec.cpp -o 4_math_novec_riscv
riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./4_math_riscv.cpp -o 4_math_riscv
riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./4_math_riscv_m1.cpp -o 4_math_riscv_m1

riscv64-unknown-linux-gnu-g++ -O2 -fopenmp -march=rv64gcv0p8 ./5_math_parallel_riscv.cpp -o 5_math_parallel_riscv
