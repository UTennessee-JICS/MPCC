# MPCC
Optimized Parallelized Matrix Pearsons Correlation Coefficient

The code presented here is an attempt to provide an algorithm to perform Pearsons Correlation Coefficient 
calculations at a large scale for data sets arranged as rows or columns in rectangular matrices. This 
particular algorithm was designed to be performant in the presence of missing data.

The initial code considers implementation on a multicore (single node) shared memory machine with thread 
level parallelism and vectorization capability. In future versions of this code, we hope to expand the 
algorithm to include distributed memory parallelism.

Initial work has been done to compute using GPUs, more information can be found HERE

### Availability

MPCC is written in C and as such can be used from any programming language which supports calling functions 
via the Foreign Function Interface (FFI). Bindings for the R language for statistical computing are 
available.

#### R language for statistical computing

MPCC can be used directly from the R language for statistical computing. To learn how to install it 
please visit the [R instructions](inst/README_R.md).

#### Using MPCC from another language

MPCC can be directly used in C/C++ projects, and other languages via the Foreign Function Interface (FFI)
please visit the [C/C++ instructions](inst/README_CPP.md) for more information.

### Tests

MPCC comes with an extensive test-suite to make sure the results are identical to correlation coefficients 
computed by e.g. the cor() function in R. The R regression test-suite can be found (here)[tests/]

### Issues

Issues can be raised through the github issue tracker.

### Contributing 

Want to contribute? Great! Contribute to the MPCC source code by forking the Github repository, 
and sending us pull requests. Its also possible to just post comments on code / commits. Or be 
a maintainer, and adopt a function

### Contact

Questions can be send via email to: Danny Arends <Danny.Arends@gmail.com>, Chad Burdyshaw <chadburdyshaw@gmail.com>, Glenn Brook <glenn-brook@tennessee.edu>
