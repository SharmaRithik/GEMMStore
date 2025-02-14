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

// Additional macros and functions for shared memory optimization
#define get_base_y_block() (blockIdx.y * THREADS_Y_VALUE)
#define get_base_x_block() (blockIdx.x * THREADS_X_VALUE)
#define get_flattened_id() (get_local_id_y() * THREADS_X_VALUE + get_local_id_x())
#define get_flattened_dim() (THREADS_X_VALUE * THREADS_Y_VALUE)

__device__ __forceinline__ void cache_f32(float *target, const int dim1, const int dim2, const float *src, const int src_dim0, const int src_dim1, const int src_stride)
{
    const uint flattened_id = get_flattened_id();
    const uint flattened_dim = get_flattened_dim();
    const uint new_j = flattened_id % dim2;
    const uint new_i = flattened_id / dim2;
    const uint stride_i = flattened_dim / dim2;
    const float *src2 = src + src_dim0 * src_stride;

    if (flattened_dim > dim1 * dim2) {
        const uint i = new_i;
        if (index2D(i, new_j, dim2) < dim1 * dim2) {
            target[index2D(i, new_j, dim2)] = src2[index2D(i, src_dim1 + new_j, src_stride)];
        }
    }
    else if (flattened_dim == dim1 * dim2) {
        const uint i = new_i;
        target[index2D(i, new_j, dim2)] = src2[index2D(i, src_dim1 + new_j, src_stride)];
    }
    else {
        for (uint outer = 0; outer < dim1 / (stride_i); outer++) {
            const uint i = outer * stride_i + new_i;
            target[index2D(i, new_j, dim2)] = src2[index2D(i, src_dim1 + new_j, src_stride)];
        }
    }
}

__global__ void foo(const float* A, const float* B, float* C, int n) {
    // Define standardized variables for global and local IDs
    const uint i = get_global_id_y();  // Global row index
    const uint j = get_global_id_x();  // Global column index
    const uint local_i = get_local_id_y();  // Local row index within the block
    const uint local_j = get_local_id_x();  // Local column index within the block

    // Initialize the accumulator for the matrix multiplication
    float value = 0;

    // Shared memory buffer for caching A
    __shared__ float A_shared[THREADS_Y_VALUE * TUNE_SPLIT_VALUE];

    // Perform the matrix multiplication with loop splitting
    for (int k_outer = 0; k_outer < n; k_outer += TUNE_SPLIT_VALUE) {
        // Cache the relevant portion of A into shared memory
        cache_f32(A_shared, THREADS_Y_VALUE, TUNE_SPLIT_VALUE, A, get_base_y_block(), k_outer, n);

        // Synchronize threads to ensure shared memory is fully populated
        __syncthreads();

        // Inner loop for computation
        for (int k_inner = 0; k_inner < TUNE_SPLIT_VALUE; k_inner++) {
            int k = k_outer + k_inner;
            value += A_shared[index2D(local_i, k_inner, TUNE_SPLIT_VALUE)] * B[index2D(k, j, n)];
        }

        // Synchronize threads before the next iteration
        __syncthreads();
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