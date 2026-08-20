#include "utils.h"

#include <cmath>
#include <cstdio>
#include <random>

void cuda_check(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA error] %s:%d: %s (%s)\n", file, line,
            cudaGetErrorString(err), cudaGetErrorName(err));
    exit(EXIT_FAILURE);
  }
}

void cublas_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "[cuBLAS error] %s:%d: status %d\n", file, line,
            static_cast<int>(status));
    exit(EXIT_FAILURE);
  }
}

void fill_random(std::vector<float> &m, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &v : m) v = dist(gen);
}

cublasHandle_t make_cublas_handle() {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // Without this cuBLAS may drop to TF32 tensor cores, which is a different
  // (lower-precision, much faster) instruction than the FP32 FFMA the kernels
  // in this repo issue.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  return handle;
}

void cublas_sgemm(cublasHandle_t handle, int N, const float *dA,
                  const float *dB, float *dC) {
  const float alpha = 1.0f, beta = 0.0f;
  // Column-major cuBLAS sees our row-major A as A^T. Asking it for
  // (B^T)(A^T) = (AB)^T, stored column-major, is byte-identical to AB stored
  // row-major -- hence dB before dA.
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                           dB, N, dA, N, &beta, dC, N));
}

ErrorReport compare(const std::vector<float> &got,
                    const std::vector<float> &ref) {
  ErrorReport r{0.0, 0.0, 0.0, 0.0, -1};
  for (size_t i = 0; i < ref.size(); ++i) {
    const double a = static_cast<double>(got[i]);
    const double b = static_cast<double>(ref[i]);
    const double diff = std::fabs(a - b);
    if (diff > r.max_abs) {
      r.max_abs = diff;
      r.bad_index = static_cast<int>(i);
    }
    r.ref_inf_norm = std::fmax(r.ref_inf_norm, std::fabs(b));
  }
  if (r.ref_inf_norm > 0.0) r.rel = r.max_abs / r.ref_inf_norm;

  // Elementwise relative error is reported too, but only over entries that are
  // not themselves the result of near-total cancellation: with uniform [-1,1)
  // inputs a handful of C entries land arbitrarily close to zero, and dividing
  // by those measures the input distribution, not the kernel.
  const double floor_mag = 1e-3 * r.ref_inf_norm;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double b = std::fabs(static_cast<double>(ref[i]));
    if (b < floor_mag) continue;
    const double e = std::fabs(static_cast<double>(got[i]) - ref[i]) / b;
    r.max_elem_rel = std::fmax(r.max_elem_rel, e);
  }
  return r;
}

void cpu_sgemm_reference(int N, const std::vector<float> &A,
                         const std::vector<float> &B, std::vector<float> &C) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < N; ++k)
        acc += static_cast<double>(A[i * N + k]) * static_cast<double>(B[k * N + j]);
      C[i * N + j] = static_cast<float>(acc);
    }
  }
}
