This kernel is a simple matrix multiplication: C = A * B where A and B
are square matrixes.

We assume that the size of the matrixes is a power of 2, and is
sufficiently large, e.g., at least (1024 x 1024).

It would be nice for the explicit assumptions to be derived, but we
don't have that immediately.

Each kernel has a set of tuning parameters, given in the .py file in
each directory.

The kernels are numbered in increasing complexity. In the case where
the kernels have similar complexity, they are abitrarily given a
character suffix, e.g., 'a' or 'b'.

Here is a brief description of each kernel, as well as it's delta with
the previous kernel.

kernel 0: the input kernel

kernel 1: simply refactor the IDs into macros

kernel 2: simply refactor 2D indexes into a macro

kernel 3: split the loop to prepare for tiling

kernel 4: tile one of matrix A or matrix B into shared memory (two kernels here)

kernel 5: tile both matrix A and B into shared memory

kernel 6: coarsen the Y dimension

kernel 7: coarsen the X and Y dimension (not sure it makes sense to coarsen only the X dimension)

kernel 8: tile one of matrix A or matrix B into registers (two kernels here)

kernel 9: tile both A and B into registers