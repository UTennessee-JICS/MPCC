# Go inside the repository and run
#$ autoconf

# Go outside the repository and run to compile

R CMD INSTALL --configure-args='--with-mkl-home=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/' MPCC

