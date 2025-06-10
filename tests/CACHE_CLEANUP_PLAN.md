# Hugging Face Cache Cleanup Plan

## Current State Analysis

### Bee1reason Models
- **beetleware repo**: 8.4GB (Q4_K_M only)
- **mradermacher repo**: 159GB (All quantizations: Q4_K_S, Q4_K_M, Q6_K, Q8_0, etc.)

### Duplication Issue
- Both repos have Q4_K_M but as different blobs (8.4GB each = 16.8GB total)
- beetleware: `4239be36dd206a4a80fe6901551a331abaa453e14459939633881709ac43a3e5`
- mradermacher: `b17a5d0aa671fe8fad0abdc9c9914386f9a5b73bc69366500ed8f99b371a4552`

## Cleanup Strategy

### 1. Keep Only Best Quantizations
From our benchmarking results, keep only:
- **Q4_K_S**: Fastest (mradermacher: `334c5a...` = 8.0GB)
- **Q4_K_M**: Balanced (choose one repo)
- **Q6_K**: Best quality (mradermacher: `00cc22...` = 12GB)

### 2. Remove Entire beetleware Repo
Since mradermacher has all quantizations including Q4_K_M, remove beetleware completely.
- Saves: 8.4GB immediately
- Eliminates duplication

### 3. Remove Unused Quantizations from mradermacher
Keep only Q4_K_S, Q4_K_M, Q6_K and remove:
- Q8_0, Q2_K, Q3_K_S, Q3_K_M, Q5_K_S, Q5_K_M, IQ variants, etc.
- Estimated savings: ~140GB

### 4. Update Model Registry
Update to use only mradermacher repo with best quantizations.

## Implementation Steps

1. Test current models work
2. Remove beetleware repo completely  
3. Remove unused quantization blobs from mradermacher
4. Update model registry
5. Test final setup

## Expected Disk Savings
- Remove beetleware: 8.4GB
- Remove unused quantizations: ~140GB
- **Total savings: ~148GB**
- **Final size: ~20GB** (Q4_K_S + Q4_K_M + Q6_K)
