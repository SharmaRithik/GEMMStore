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

// New macros for coarsening
#define THREADS_Y_DIM (THREADS_Y_VALUE * COARSEN_Y_VALUE)
#define THREADS_X_DIM (THREADS_X_VALUE * COARSEN_X_VALUE)

#define get_base_y_block_coarsened() (blockIdx.y * THREADS_Y_VALUE * COARSEN_Y_VALUE)
#define get_base_x_block_coarsened() (blockIdx.x * THREADS_X_VALUE * COARSEN_X_VALUE)

#define get_global_id_y_coarsened(coarsen_y) ((blockIdx.y * THREADS_Y_VALUE * COARSEN_Y_VALUE + get_local_id_y()) + THREADS_Y_VALUE * coarsen_y)
#define get_global_id_x_coarsened(coarsen_x) ((blockIdx.x * THREADS_X_VALUE * COARSEN_X_VALUE + get_local_id_x()) + THREADS_X_VALUE * coarsen_x)

#define get_local_id_y_coarsened(coarsen_y) (get_local_id_y() + THREADS_Y_VALUE * coarsen_y)
#define get_local_id_x_coarsened(coarsen_x) (get_local_id_x() + THREADS_X_VALUE * coarsen_x)

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

    // Declare an array to store coarsened accumulators
    float accumulator_array[COARSEN_Y_VALUE * COARSEN_X_VALUE] = {0};

    // Shared memory buffer for caching A
    __shared__ float A_shared[THREADS_Y_DIM * TUNE_SPLIT_VALUE];

    // Shared memory buffer for caching B
    __shared__ float B_shared[TUNE_SPLIT_VALUE * THREADS_X_DIM];

    // Register cache for A_shared
    float A_registers[COARSEN_Y_VALUE] = {0.0};

    // Register cache for B_shared
    float B_registers[COARSEN_X_VALUE] = {0.0};

    // Perform the matrix multiplication with loop splitting
    for (int k_outer = 0; k_outer < n; k_outer += TUNE_SPLIT_VALUE) {
        // Cache the relevant portion of A into shared memory
        cache_f32(A_shared, THREADS_Y_DIM, TUNE_SPLIT_VALUE, A, get_base_y_block_coarsened(), k_outer, n);

        // Cache the relevant portion of B into shared memory
        cache_f32(B_shared, TUNE_SPLIT_VALUE, THREADS_X_DIM, B, k_outer, get_base_x_block_coarsened(), n);

        // Synchronize threads to ensure shared memory is fully populated
        __syncthreads();

        // Inner loop for computation
        for (int k_inner = 0; k_inner < TUNE_SPLIT_VALUE; k_inner++) {
            // Cache A_shared into registers
            for (int coarsen_y = 0; coarsen_y < COARSEN_Y_VALUE; coarsen_y++) {
                const uint coarsened_local_i = get_local_id_y_coarsened(coarsen_y);
                A_registers[coarsen_y] = A_shared[index2D(coarsened_local_i, k_inner, TUNE_SPLIT_VALUE)];
            }

            // Cache B_shared into registers
            for (int coarsen_x = 0; coarsen_x < COARSEN_X_VALUE; coarsen_x++) {
                const uint coarsened_local_j = get_local_id_x_coarsened(coarsen_x);
                B_registers[coarsen_x] = B_shared[index2D(k_inner, coarsened_local_j, THREADS_X_DIM)];
            }

            // Coarsen Y and X Accumulate transformation applied here
            for (int coarsen_x = 0; coarsen_x < COARSEN_X_VALUE; coarsen_x++) {
                for (int coarsen_y = 0; coarsen_y < COARSEN_Y_VALUE; coarsen_y++) {
                    // Accumulate the result
                    accumulator_array[index2D(coarsen_y, coarsen_x, COARSEN_X_VALUE)] += 
                        A_registers[coarsen_y] * 
                        B_registers[coarsen_x];
                }
            }
        }

        // Synchronize threads before the next iteration
        __syncthreads();
    }

    // Write the result to the output matrix
    for (int coarsen_x = 0; coarsen_x < COARSEN_X_VALUE; coarsen_x++) {
        const uint coarsened_j = get_global_id_x_coarsened(coarsen_x);
        for (int coarsen_y = 0; coarsen_y < COARSEN_Y_VALUE; coarsen_y++) {
            const uint coarsened_i = get_global_id_y_coarsened(coarsen_y);
            C[index2D(coarsened_i, coarsened_j, n)] = accumulator_array[index2D(coarsen_y, coarsen_x, COARSEN_X_VALUE)];
        }
    }
}

/*KERNEL END*/

/*HOST BEGIN*/

#ifdef INCLUDE_HOST
void foo_host(const float* A, const float* B, float* C, int n) {
    // Define the block and grid dimensions
    dim3 blockDim(THREADS_X_VALUE, THREADS_Y_VALUE);
    dim3 gridDim((n / blockDim.x) / COARSEN_X_VALUE, (n / blockDim.y) / COARSEN_Y_VALUE);

    // Launch the kernel
    foo<<<gridDim, blockDim>>>(A, B, C, n);
}
#endif
/*HOST END*/