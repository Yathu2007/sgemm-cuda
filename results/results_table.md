### GFLOPS

| Kernel | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `cublas` | 518.1 | 1927.5 | 4096.0 | 5811.1 | 6379.6 | 8009.6 |

### % of cuBLAS

| Kernel | N=128 | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|---|
| `cublas` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

### Median time (ms) and max relative error vs cuBLAS

| Kernel | N | median ms | GFLOPS | % cuBLAS | max rel err |
|---|---|---|---|---|---|
| `cublas` | 128 | 0.0081 | 518.1 | 100.0% | 0.00e+00 |
| `cublas` | 256 | 0.0174 | 1927.5 | 100.0% | 0.00e+00 |
| `cublas` | 512 | 0.0655 | 4096.0 | 100.0% | 0.00e+00 |
| `cublas` | 1024 | 0.3696 | 5811.1 | 100.0% | 0.00e+00 |
| `cublas` | 2048 | 2.6929 | 6379.6 | 100.0% | 0.00e+00 |
| `cublas` | 4096 | 17.1592 | 8009.6 | 100.0% | 0.00e+00 |

### Observed GPU state during the sweep

4 samples at 1 Hz.

| | min | median | max |
|---|---|---|---|
| SM clock (MHz) | 690 | 2055 | 2715 |
| Power (W) | 7.0 | 16.1 | 93.7 |
| Temperature (C) | 45 | 49 | 52 |
