#pragma once

#include <cublas_v2.h>

#include <string>
#include <vector>

// KERNEL_CUBLAS is the baseline every rung will be measured against. Ladder
// kernels take ids 0, 1, ... as they land; KERNEL_MAX_ID bounds the ids the
// CLI accepts, so it stays at -1 while the ladder is still empty.
constexpr int KERNEL_CUBLAS = -1;
constexpr int KERNEL_MAX_ID = -1;  // bump as rungs land

// Returns the ladder id for "0".."5" or "cublas"; INT_MIN if unrecognised.
int parse_kernel_id(const std::string &s);
const char *kernel_name(int id);

// Single launch of the selected kernel. Does not synchronise.
void run_kernel(int id, int N, const float *dA, const float *dB, float *dC,
                cublasHandle_t handle);

struct BenchResult {
  double median_ms;
  double min_ms;
  double p95_ms;
  int iters;
};

// cudaEvent-timed benchmark of run_kernel(). The events bracket the kernel
// launch only -- allocation and H2D/D2H are outside. `warmup` iterations are
// discarded before the `iters` timed ones so the first-launch cost (module
// load, cuBLAS heuristic selection, clock ramp) never enters the sample.
BenchResult benchmark(int id, int N, const float *dA, const float *dB,
                      float *dC, cublasHandle_t handle, int warmup, int iters);

inline double gflops(int N, double ms) {
  return (2.0 * N * N * N) / (ms * 1e-3) / 1e9;
}
