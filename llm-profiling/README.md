# Profiling Analysis: Llama 3.2 1B q0f16 on Apple M2

## Top 10 Most Time-Consuming Kernels

| Kernel Name                      | Peak Time (ms) |
|----------------------------------|---------------|
| NT_matmul7_kernel_2              | 29.62         |
| NT_matmul8_kernel_2              | 13.75         |
| NT_matmul5_kernel_2              | 11.79         |
| NT_matmul6_kernel_2              | 8.12          |
| NT_matmul12_kernel               | 1.12          |
| NT_matmul14_kernel               | 8.41          |
| batch_prefill_ragged_kv_kernel   | 6.40          |
| NT_matmul13_kernel               | 0.60          |
| batch_prefill_paged_kv_kernel    | 1.61          |
| NT_matmul10_kernel               | 0.25          |

---

## Kernel Execution Statistics (Sorted by Total Time)

| Kernel Name                            | Total Time (ms) | Iterations | Avg Time (ms) | Peak Time (ms) | % of Total | Cumulative % |
|----------------------------------------|----------------|------------|--------------|---------------|------------|--------------|
| NT_matmul7_kernel_2                    | 975.35         | 63         | 15.48        | 29.62         | 41.59%     | 41.59%       |
| NT_matmul8_kernel_2                    | 542.74         | 64         | 8.48         | 13.75         | 23.15%     | 64.74%       |
| NT_matmul5_kernel_2                    | 198.00         | 64         | 3.09         | 11.79         | 8.44%      | 73.18%       |
| NT_matmul6_kernel_2                    | 144.76         | 62         | 2.33         | 8.12          | 6.17%      | 79.36%       |
| NT_matmul12_kernel                     | 133.16         | 126        | 1.06         | 1.12          | 5.68%      | 85.03%       |
| NT_matmul14_kernel                     | 80.82          | 11         | 7.35         | 8.41          | 3.45%      | 88.48%       |
| batch_prefill_ragged_kv_kernel         | 79.11          | 63         | 1.26         | 6.40          | 3.37%      | 91.86%       |
| NT_matmul13_kernel                     | 67.06          | 125        | 0.54         | 0.60          | 2.86%      | 94.71%       |
| batch_prefill_paged_kv_kernel          | 31.23          | 30         | 1.04         | 1.61          | 1.33%      | 96.05%       |
| NT_matmul10_kernel                     | 26.83          | 125        | 0.21         | 0.25          | 1.14%      | 97.19%       |
| batch_decode_paged_kv_kernel           | 24.40          | 126        | 0.19         | 0.21          | 1.04%      | 98.23%       |
| NT_matmul11_kernel                     | 19.27          | 126        | 0.15         | 0.19          | 0.82%      | 99.05%       |
| fuse_add_norm_prefill_kernel           | 9.48           | 375        | 0.03         | 0.08          | 0.40%      | 99.46%       |
| fused_split1_silu1_multiply1_kernel    | 5.11           | 63         | 0.08         | 0.14          | 0.22%      | 99.67%       |
| fused_rope_kernel                      | 3.91           | 191        | 0.02         | 0.16          | 0.17%      | 99.84%       |
| tir_kv_cache_transpose_append_kernel   | 1.72           | 189        | 0.01         | 0.02          | 0.07%      | 99.92%       |
| fused_split2_silu2_multiply2_kernel    | 1.03           | 125        | 0.01         | 0.01          | 0.04%      | 99.96%       |
| merge_state_inplace_kernel             | 0.32           | 32         | 0.01         | 0.02          | 0.01%      | 99.97%       |
| rms_norm1_kernel                       | 0.25           | 4          | 0.06         | 0.09          | 0.01%      | 99.98%       |
| rms_norm2_kernel                       | 0.21           | 8          | 0.03         | 0.03          | 0.01%      | 99.99%       |
| take1_kernel                           | 0.17           | 12         | 0.01         | 0.04          | 0.01%      | 100.00%      |
| index_kernel                           | 0.02           | 4          | 0.00         | 0.00          | 0.00%      | 100.00%      |

---

