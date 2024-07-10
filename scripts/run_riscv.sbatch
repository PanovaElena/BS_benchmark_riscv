#!/bin/bash

#SBATCH -N 1
#SBATCH -t 120

export KMP_AFFINITY="granularity=fine,compact,1,0"

N=300000000
NP=3

echo "--- baseline_AoS_riscv ---"
./baseline_AoS_riscv $N $NP
./baseline_AoS_riscv $N $NP
./baseline_AoS_riscv $N $NP

echo "--- baseline_SoA_riscv ---"
./baseline_SoA_riscv $N $NP
./baseline_SoA_riscv $N $NP
./baseline_SoA_riscv $N $NP

echo "--- 1_invariant_riscv ---"
./1_invariant_riscv $N $NP
./1_invariant_riscv $N $NP
./1_invariant_riscv $N $NP

echo "--- 2_parallel_riscv ---"
for thr in 1 2 4
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_riscv $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_riscv $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_riscv $N $NP
    unset OMP_NUM_THREADS
done

echo "--- 2_parallel_numa_riscv ---"
for thr in 1 2 4
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_numa_riscv $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_numa_riscv $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_numa_riscv $N $NP
    unset OMP_NUM_THREADS
done

echo "--- 3_vectorization_pragma_simd_riscv ---"
./3_vectorization_pragma_simd_riscv $N $NP
./3_vectorization_pragma_simd_riscv $N $NP
./3_vectorization_pragma_simd_riscv $N $NP

echo "--- 3_vectorization_intrinsic_riscv ---"
./3_vectorization_intrinsic_riscv $N $NP
./3_vectorization_intrinsic_riscv $N $NP
./3_vectorization_intrinsic_riscv $N $NP

echo "--- 4_math_novec_riscv ---"
./4_math_novec_riscv $N $NP
./4_math_novec_riscv $N $NP
./4_math_novec_riscv $N $NP

echo "--- 4_math_riscv_m1 ---"
./4_math_riscv_m1 $N $NP
./4_math_riscv_m1 $N $NP
./4_math_riscv_m1 $N $NP

echo "--- 4_math_riscv ---"
./4_math_riscv $N $NP
./4_math_riscv $N $NP
./4_math_riscv $N $NP

