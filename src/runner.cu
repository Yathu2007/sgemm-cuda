#include "runner.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>

#include "kernels/00_naive.cuh"
#include "kernels/01_coalesced.cuh"
#include "utils.h"

int parse_kernel_id(const std::string &s) {
  if (s == "cublas" || s == "CUBLAS") return KERNEL_CUBLAS;
  try {
    size_t pos = 0;
    const int v = std::stoi(s, &pos);
    if (pos != s.size() || v < 0 || v > KERNEL_MAX_ID) return INT_MIN;
    return v;
  } catch (...) {
    return INT_MIN;
  }
}

const char *kernel_name(int id) {
  switch (id) {
    case KERNEL_CUBLAS: return "cublas";
    case 0: return "K0_naive";
    case 1: return "K1_coalesced";
    default: return "unknown";
  }
}

void run_kernel(int id, int N, const float *dA, const float *dB, float *dC,
                cublasHandle_t handle) {
  switch (id) {
    case KERNEL_CUBLAS: cublas_sgemm(handle, N, dA, dB, dC); break;
    case 0: k0::launch(N, dA, dB, dC); break;
    case 1: k1::launch(N, dA, dB, dC); break;
    default:
      fprintf(stderr, "run_kernel: unknown kernel id %d\n", id);
      exit(EXIT_FAILURE);
  }
}

BenchResult benchmark(int id, int N, const float *dA, const float *dB,
                      float *dC, cublasHandle_t handle, int warmup, int iters) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) run_kernel(id, N, dA, dB, dC, handle);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    run_kernel(id, N, dA, dB, dC, handle);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::sort(samples.begin(), samples.end());
  BenchResult r;
  r.iters = iters;
  r.min_ms = samples.front();
  // Median over an even count averages the two central samples.
  const size_t n = samples.size();
  r.median_ms = (n % 2) ? samples[n / 2]
                        : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
  // Nearest-rank p95, so the reported tail is an actual observed sample.
  size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * n)) - 1;
  if (p95_idx >= n) p95_idx = n - 1;
  r.p95_ms = samples[p95_idx];

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return r;
}
