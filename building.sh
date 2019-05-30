export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2018/linux/mkl/lib/intel64/:/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/lib/intel64

R CMD INSTALL --configure-args='--with-mpcc-include=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/include/ --with-mpcc-lib=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64/' MPCC


