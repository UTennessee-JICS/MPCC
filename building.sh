# Install MKL and export MKL_HOME
# export MKL_HOME=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/

# Go inside the repository and run
#$ autoconf

# Go outside the repository and run to compile
R CMD INSTALL --configure-args='--with-mkl-home=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/' MPCC

# Check the R package, should first export MKL_HOME
# rm -f MPCC/config.* MPCC/src/*.o MPCC/src/*.so MPCC/src/Makevars MPCC/src/config.h
# R CMD check MPCC
