#!/bin/bash
#SBATCH --job-name=MPCC
#SBATCH --constrain=skylake
#SBATCH -N 1 # number of nodes
#SBATCH --exclusive
#SBATCH --time 00:40:00 # time (D-HH:MM)
#SBATCH --output MPCC.out # STDOUT

export OMP_NUM_THREADS=40
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MKL_NUM_THREADS=40
export MKL_ENABLE_INSTRUCTIONS=AVX512
module swap intel-compilers intel-compilers/latest
#cat /proc/cpuinfo 
./MPCC

