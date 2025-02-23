## Profiling Analysis: Llama 3.2 1B q0f16 on Apple M2

<small>

### Top 10 Most Time-Consuming Kernels

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

### Kernel Execution Statistics (Sorted by Total Time)

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

</small>

---

## Profiling Analysis: Llama 3.2 1B q4f16_1 on Apple M2

<small>

### Top 10 Most Time-Consuming Kernels

Kernel Name                             | Peak_Time_ms   
---------------------------------------------------------
fused_dequantize3_NT_matmul7_kernel_2   | 40.57          
fused_dequantize4_NT_matmul8_kernel_2   | 10.85          
fused_dequantize1_NT_matmul5_kernel_2   | 12.42          
fused_dequantize2_NT_matmul6_kernel_2   | 8.41           
fused_dequantize3_NT_matmul12_kernel    | 0.78           
batch_prefill_ragged_kv_kernel          | 5.91           
fused_dequantize_NT_matmul14_kernel     | 6.46           
fused_dequantize4_NT_matmul13_kernel    | 0.62           
batch_prefill_paged_kv_kernel           | 1.61           
batch_decode_paged_kv_kernel            | 0.20  

---

### Kernel Execution Statistics (Sorted by Total Time)

Kernel Name                             | Total_Time_ms   | Iterations | Avg_Time_ms     | Peak_Time_ms    | % of Total   | Cumulative %
----------------------------------------------------------------------------------------------------------------------------------------
fused_dequantize3_NT_matmul7_kernel_2   | 1030.28         | 64         | 16.10           | 40.57           | 45.64%       | 45.64%      
fused_dequantize4_NT_matmul8_kernel_2   | 518.01          | 62         | 8.36            | 10.85           | 22.95%       | 68.58%      
fused_dequantize1_NT_matmul5_kernel_2   | 209.64          | 63         | 3.33            | 12.42           | 9.29%        | 77.87%      
fused_dequantize2_NT_matmul6_kernel_2   | 139.54          | 61         | 2.29            | 8.41            | 6.18%        | 84.05%      
fused_dequantize3_NT_matmul12_kernel    | 83.42           | 125        | 0.67            | 0.78            | 3.70%        | 87.74%      
batch_prefill_ragged_kv_kernel          | 80.05           | 64         | 1.25            | 5.91            | 3.55%        | 91.29%      
fused_dequantize_NT_matmul14_kernel     | 58.71           | 11         | 5.34            | 6.46            | 2.60%        | 93.89%      
fused_dequantize4_NT_matmul13_kernel    | 36.86           | 121        | 0.30            | 0.62            | 1.63%        | 95.52%      
batch_prefill_paged_kv_kernel           | 30.56           | 31         | 0.99            | 1.61            | 1.35%        | 96.87%      
batch_decode_paged_kv_kernel            | 20.95           | 126        | 0.17            | 0.20            | 0.93%        | 97.80%      
fused_dequantize1_NT_matmul10_kernel    | 16.86           | 126        | 0.13            | 0.16            | 0.75%        | 98.55%      
fused_dequantize2_NT_matmul11_kernel    | 11.54           | 123        | 0.09            | 0.12            | 0.51%        | 99.06%      
fuse_add_norm_prefill_kernel            | 8.74            | 373        | 0.02            | 0.08            | 0.39%        | 99.45%      
fused_split1_silu1_multiply1_kernel     | 5.29            | 62         | 0.09            | 0.16            | 0.23%        | 99.68%      
fused_rope_kernel                       | 3.60            | 185        | 0.02            | 0.16            | 0.16%        | 99.84%      
tir_kv_cache_transpose_append_kernel    | 1.70            | 186        | 0.01            | 0.04            | 0.08%        | 99.92%      
fused_split2_silu2_multiply2_kernel     | 0.92            | 125        | 0.01            | 0.01            | 0.04%        | 99.96%      
merge_state_inplace_kernel              | 0.30            | 31         | 0.01            | 0.01            | 0.01%        | 99.97%      
rms_norm1_kernel                        | 0.23            | 4          | 0.06            | 0.08            | 0.01%        | 99.98%      
rms_norm2_kernel                        | 0.22            | 8          | 0.03            | 0.03            | 0.01%        | 99.99%      
fused_dequantize_take1_kernel           | 0.20            | 12         | 0.02            | 0.05            | 0.01%        | 100.00%     
index_kernel                            | 0.02            | 4          | 0.00            | 0.01            | 0.00%        | 100.00%

</small>

---

## Profiling Analysis: Llama 3.2 1B q4f32_1 on Apple M2

<small>

### Top 10 Most Time-Consuming Kernels

Kernel Name                                        | Peak_Time_ms   
--------------------------------------------------------------------
fused_dequantize3_NT_matmul7_kernel_2              | 53.14          
fused_dequantize4_fused_NT_matmul8_add1_kernel_2   | 22.82          
fused_dequantize1_NT_matmul5_kernel_2              | 23.85          
fused_dequantize2_fused_NT_matmul6_add1_kernel_2   | 8.43           
batch_prefill_ragged_kv_kernel                     | 6.09           
fused_dequantize3_NT_matmul12_kernel               | 1.05           
fused_dequantize_NT_matmul14_kernel                | 8.14           
fused_dequantize4_fused_NT_matmul13_add2_kernel    | 0.53           
batch_prefill_paged_kv_kernel                      | 2.18           
fused_dequantize1_NT_matmul10_kernel               | 0.21 

---

### Kernel Execution Statistics (Sorted by Total Time)

Kernel Name                                        | Total_Time_ms   | Iterations | Avg_Time_ms     | Peak_Time_ms    | % of Total   | Cumulative %
---------------------------------------------------------------------------------------------------------------------------------------------------
fused_dequantize3_NT_matmul7_kernel_2              | 1928.01         | 63         | 30.60           | 53.14           | 48.02%       | 48.02%      
fused_dequantize4_fused_NT_matmul8_add1_kernel_2   | 991.04          | 62         | 15.98           | 22.82           | 24.68%       | 72.70%      
fused_dequantize1_NT_matmul5_kernel_2              | 421.65          | 64         | 6.59            | 23.85           | 10.50%       | 83.20%      
fused_dequantize2_fused_NT_matmul6_add1_kernel_2   | 268.05          | 62         | 4.32            | 8.43            | 6.68%        | 89.88%      
batch_prefill_ragged_kv_kernel                     | 134.73          | 63         | 2.14            | 6.09            | 3.36%        | 93.23%      
fused_dequantize3_NT_matmul12_kernel               | 77.97           | 125        | 0.62            | 1.05            | 1.94%        | 95.18%      
fused_dequantize_NT_matmul14_kernel                | 47.70           | 11         | 4.34            | 8.14            | 1.19%        | 96.36%      
fused_dequantize4_fused_NT_matmul13_add2_kernel    | 38.76           | 124        | 0.31            | 0.53            | 0.97%        | 97.33%      
batch_prefill_paged_kv_kernel                      | 38.68           | 32         | 1.21            | 2.18            | 0.96%        | 98.29%      
fused_dequantize1_NT_matmul10_kernel               | 15.69           | 126        | 0.12            | 0.21            | 0.39%        | 98.68%      
batch_decode_paged_kv_kernel                       | 15.13           | 124        | 0.12            | 0.21            | 0.38%        | 99.06%      
fused_dequantize2_fused_NT_matmul11_add2_kernel    | 10.85           | 126        | 0.09            | 0.14            | 0.27%        | 99.33%      
fused_split1_silu1_multiply1_kernel                | 9.74            | 64         | 0.15            | 0.28            | 0.24%        | 99.57%      
rms_norm2_kernel                                   | 5.66            | 258        | 0.02            | 0.03            | 0.14%        | 99.71%      
rms_norm1_kernel                                   | 4.98            | 130        | 0.04            | 0.11            | 0.12%        | 99.84%      
fused_rope_kernel                                  | 3.27            | 187        | 0.02            | 0.10            | 0.08%        | 99.92%      
tir_kv_cache_transpose_append_kernel               | 2.00            | 188        | 0.01            | 0.04            | 0.05%        | 99.97%      
fused_split2_silu2_multiply2_kernel                | 0.70            | 124        | 0.01            | 0.01            | 0.02%        | 99.99%      
merge_state_inplace_kernel                         | 0.37            | 32         | 0.01            | 0.02            | 0.01%        | 100.00%     
fused_dequantize_take1_kernel                      | 0.17            | 12         | 0.01            | 0.05            | 0.00%        | 100.00%     
index_kernel                                       | 0.01            | 4          | 0.00            | 0.00            | 0.00%        | 100.00%  

</small>

---
