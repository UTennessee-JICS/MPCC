R CMD INSTALL --configure-args='--with-mpcc-include=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/include/ --with-mpcc-lib=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/' MPCC

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/
