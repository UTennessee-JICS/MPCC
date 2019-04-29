# MPCC
Optimized Parallelized Matrix Pearsons Correlation Coefficient

The code presented here is an attempt to provide an algorithm to perform Pearsons Correlation Coefficient calculations at a large scale for data sets arranged as rows or columns in rectangular matrices. This particular algorithm was designed to be performant in the presence of missing data.

The initial code considers implementation on a multicore (single node) shared memory machine with thread level parallelism and vectorization capability.
In future versions of this code, we hope to expand the algorithm to include distributed memory parallelism.
