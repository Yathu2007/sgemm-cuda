#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
void cuda_check(cudaError_t err, const char *file, int line);
void cublas_check(cublasStatus_t status, const char *file, int line);

#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)
#define CUBLAS_CHECK(x) cublas_check((x), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Host-side setup
// ---------------------------------------------------------------------------

// Fixed-seed uniform [-1, 1) fill, so every run of the harness sees identical
// inputs and results are reproducible across machines and sessions.
void fill_random(std::vector<float> &m, unsigned seed);

// ---------------------------------------------------------------------------
// cuBLAS reference
// ---------------------------------------------------------------------------

// Row-major C = A * B via cuBLAS, which is column-major. Computing C^T = B^T *
// A^T with both operands CUBLAS_OP_N and ld = N gives exactly the row-major
// product, so the operands are passed in swapped order (B first, then A).
// The handle must have been created with make_cublas_handle(), which pins the
// math mode to true FP32 (no TF32 tensor cores).
void cublas_sgemm(cublasHandle_t handle, int N, const float *dA,
                  const float *dB, float *dC);

cublasHandle_t make_cublas_handle();

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

struct ErrorReport {
  double max_abs;       // max |c - ref|
  double ref_inf_norm;  // max |ref|
  double rel;           // max_abs / ref_inf_norm  (the asserted quantity)
  double max_elem_rel;  // max |c-ref|/|ref| over entries above the noise floor
  int bad_index;        // index attaining max_abs, -1 if sizes are zero
};

// Compares two row-major N*N matrices. Bitwise equality is impossible: a
// different summation order changes FP32 rounding, so this measures the
// normwise relative error (max elementwise deviation scaled by the largest
// reference magnitude) rather than demanding exact agreement.
ErrorReport compare(const std::vector<float> &got,
                    const std::vector<float> &ref);

// Naive triple loop in double precision. Only used to sanity-check the cuBLAS
// operand swap at small N before anything downstream is trusted.
void cpu_sgemm_reference(int N, const std::vector<float> &A,
                         const std::vector<float> &B, std::vector<float> &C);
