#!/bin/bash
#SBATCH --job-name=MPCC
#SBATCH --constrain=skylake
#SBATCH -N 1 # number of nodes
#SBATCH --exclusive
#SBATCH --time 20:00:00 # time (D-HH:MM)
#SBATCH --output MPCC.out # STDOUT

#module swap PE-intel PE-gnu
#module load gcc
export CC=icc

export OMP_NUM_THREADS=40
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MKL_NUM_THREADS=40
export MKL_ENABLE_INSTRUCTIONS=AVX512

module unload gcc
module load r/3.5.0
make clean
autoconf
cd ..
R CMD INSTALL --configure-args='--with-mkl-home=$MKLROOT' MPCC
cd ./MPCC
Rscript inst/makeplot.R

