#!/bin/bash

#SBATCH -N 1
#SBATCH -t 60

export KMP_AFFINITY="granularity=fine,compact,1,0"

N=300000000
NP=3

echo "--- baseline_AoS_x86 ---"
./baseline_AoS_x86 $N $NP
./baseline_AoS_x86 $N $NP
./baseline_AoS_x86 $N $NP

echo "--- baseline_SoA_x86 ---"
./baseline_SoA_x86 $N $NP
./baseline_SoA_x86 $N $NP
./baseline_SoA_x86 $N $NP

echo "--- 1_invariant_x86 ---"
./1_invariant_x86 $N $NP
./1_invariant_x86 $N $NP
./1_invariant_x86 $N $NP

echo "--- 2_parallel_x86 ---"
for thr in 1 2 4 8 10 16 20 40
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_x86 $N $NP
    unset OMP_NUM_THREADS
done

echo "--- 2_parallel_numa_x86 ---"
for thr in 1 2 4 8 10 16 20 40
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_numa_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_numa_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./2_parallel_numa_x86 $N $NP
    unset OMP_NUM_THREADS
done

echo "--- 3_vectorization_pragma_simd_x86 ---"
./3_vectorization_pragma_simd_x86 $N $NP
./3_vectorization_pragma_simd_x86 $N $NP
./3_vectorization_pragma_simd_x86 $N $NP

echo "--- 4_math_novec_x86 ---"
./4_math_novec_x86 $N $NP
./4_math_novec_x86 $N $NP
./4_math_novec_x86 $N $NP

echo "--- 4_math_x86 ---"
./4_math_x86 $N $NP
./4_math_x86 $N $NP
./4_math_x86 $N $NP

echo "--- 5_math_parallel_x86 ---"
for thr in 1 2 4 8 10 16 20 40
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./5_math_parallel_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./5_math_parallel_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./5_math_parallel_x86 $N $NP
    unset OMP_NUM_THREADS
done

echo "--- 6_parallel_simd_x86 ---"
for thr in 1 2 4 8 10 16 20 40
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./6_parallel_simd_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./6_parallel_simd_x86 $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./6_parallel_simd_x86 $N $NP
    unset OMP_NUM_THREADS
done

echo "--- 6_parallel_simd_x86_low_accuracy ---"
for thr in 1 2 4 8 10 16 20 40
do
    export OMP_NUM_THREADS=$thr
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    numactl --physcpubind=0-"$(($thr-1))" ./6_parallel_simd_x86_low_accuracy $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./6_parallel_simd_x86_low_accuracy $N $NP
    numactl --physcpubind=0-"$(($thr-1))" ./6_parallel_simd_x86_low_accuracy $N $NP
    unset OMP_NUM_THREADS
done
