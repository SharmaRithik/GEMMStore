#include "macros.h"
/*KERNEL BEGIN*/

__global__ void foo(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;
    for (int k = 0; k < n; ++k) {
        value += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = value;
}

/*KERNEL END*/
/*HOST BEGIN*/

#ifdef INCLUDE_HOST
void foo_host(const float* A, const float* B, float* C, int n) {
    dim3 blockDim(THREADS_X_VALUE, THREADS_Y_VALUE);
    dim3 gridDim(n / blockDim.x, n / blockDim.y);
    foo<<<gridDim, blockDim>>>(A, B, C, n);
}
#endif

/*HOST END*/