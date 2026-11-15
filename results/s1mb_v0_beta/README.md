# S1MB English v1 Evaluation Receipts (Nirnay-0.5B v0-beta)

This directory contains the complete, raw benchmark receipts for all 137 tasks in the [System One Mosaic Benchmark (S1MB)](https://github.com/hotchpotch/S1MB) evaluated on `nirnay-0.5b` (`v0-beta`, checkpoint step 19,785).

## Verification
You can independently validate all 137 result files using the official S1MB validator:
```bash
uv run s1mb validate results/s1mb_v0_beta
```

## Summary Metrics
- **Official S1MB Task Avg:** 7.17
- **Noul Score (59 benchmarks):** 10.77
- **Choice Score (57 benchmarks):** 10.26
- **Score Task (21 benchmarks):** 0.48
- **Total Cases Evaluated:** 14,009
- **Failed Requests:** 0
- **Retries:** 0
