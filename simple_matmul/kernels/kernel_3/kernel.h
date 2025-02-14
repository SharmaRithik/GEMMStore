#include "macros.h"

/*KERNEL BEGIN*/

// Define macros for local and global ID calculations
#define get_local_id_x() (threadIdx.x)
#define get_local_id_y() (threadIdx.y)
#define get_local_id_z() (threadIdx.z)

#define get_global_id_x() (blockIdx.x * THREADS_X_VALUE + get_local_id_x())
#define get_global_id_y() (blockIdx.y * THREADS_Y_VALUE + get_local_id_y())
#define get_global_id_z() (blockIdx.z * THREADS_Z_VALUE + get_local_id_z())

// Define the index2D macro for 2D indexing in a 1D array
#define index2D(i, j, n) ((i) * (n) + (j))

__global__ void foo(const float* A, const float* B, float* C, int n) {
    // Define standardized variables for global and local IDs
    const uint i = get_global_id_y();  // Global row index
    const uint j = get_global_id_x();  // Global column index
    const uint local_i = get_local_id_y();  // Local row index within the block
    const uint local_j = get_local_id_x();  // Local column index within the block

    // Initialize the accumulator for the matrix multiplication
    float value = 0;

    // Perform the matrix multiplication with loop splitting
    for (int k_outer = 0; k_outer < n; k_outer += TUNE_SPLIT_VALUE) {
        for (int k_inner = 0; k_inner < TUNE_SPLIT_VALUE; k_inner++) {
            int k = k_outer + k_inner;
            value += A[index2D(i, k, n)] * B[index2D(k, j, n)];
        }
    }

    // Write the result to the output matrix
    C[index2D(i, j, n)] = value;
}

/*KERNEL END*/

/*HOST BEGIN*/

#ifdef INCLUDE_HOST
void foo_host(const float* A, const float* B, float* C, int n) {
    // Define the block and grid dimensions
    dim3 blockDim(THREADS_X_VALUE, THREADS_Y_VALUE);
    dim3 gridDim(n / blockDim.x, n / blockDim.y);

    // Launch the kernel
    foo<<<gridDim, blockDim>>>(A, B, C, n);
}
#endif
/*HOST END*/