// Above N~1650 both kernels in this repo match cuBLAS bit for bit, which looks
// like a broken correctness check. It is not: cuBLAS uses a split-k reduction
// below that size (extra parallelism it needs to fill 20 SMs) and a single
// sequential-k accumulator above it, and the sequential order is exactly what
// the naive loop does.
//
// Split-k sums in a tree, and a tree is *more accurate* than a sequential
// chain. So the hypothesis is falsifiable without a profiler: compare both
// against a float64 host reference and see whether cuBLAS is the more accurate
// of the two below the knee and indistinguishable above it.
//
//   usage: ./accuracy_probe [N] [rows]
//   rows  how many rows of C to check on the host (default 64). The host loop
//         is O(rows * N^2) in double, so keep it small.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x)                                                            \
  do {                                                                   \
    cudaError_t e = (x);                                                 \
    if (e != cudaSuccess) {                                              \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                           \
    }                                                                    \
  } while (0)
#define BK(x)                                                     \
  do {                                                            \
    cublasStatus_t s = (x);                                       \
    if (s != CUBLAS_STATUS_SUCCESS) {                             \
      printf("cuBLAS %s:%d status %d\n", __FILE__, __LINE__, (int)s); \
      exit(1);                                                    \
    }                                                             \
  } while (0)

// Same arithmetic and same k order as kernels/00_naive.cuh - kept local so the
// probe stays a standalone tool.
__global__ void sequential_k(int N, const float *A, const float *B, float *C) {
  unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < (unsigned)N && c < (unsigned)N) {
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += A[r * N + k] * B[k * N + c];
    C[r * N + c] = acc;
  }
}

static void fill(std::vector<float> &m, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (auto &v : m) v = d(g);
}

int main(int argc, char **argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 2048;
  const int rows = argc > 2 ? atoi(argv[2]) : 64;
  if (N <= 0 || rows <= 0 || rows > N) {
    printf("usage: %s [N] [rows<=N]\n", argv[0]);
    return 1;
  }
  const size_t elems = (size_t)N * N, bytes = elems * sizeof(float);

  std::vector<float> hA(elems), hB(elems), hSeq(elems), hRef(elems);
  fill(hA, 1234u);  // same seeds as the benchmark harness
  fill(hB, 1235u);

  float *dA, *dB, *dSeq, *dRef;
  CK(cudaMalloc(&dA, bytes));
  CK(cudaMalloc(&dB, bytes));
  CK(cudaMalloc(&dSeq, bytes));
  CK(cudaMalloc(&dRef, bytes));
  CK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  cublasHandle_t h;
  BK(cublasCreate(&h));
  BK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));  // no TF32
  const float alpha = 1.0f, beta = 0.0f;
  // Column-major swap, as in src/utils.cu.
  BK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N,
                 &beta, dRef, N));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));

  dim3 blk(32, 32), grd((N + 31) / 32, (N + 31) / 32);
  sequential_k<<<grd, blk>>>(N, dA, dB, dSeq);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());
  CK(cudaMemcpy(hSeq.data(), dSeq, bytes, cudaMemcpyDeviceToHost));

  // Is the GPU output even well-formed? A NaN- or zero-filled result would
  // also produce a flattering error number, so rule that out explicitly.
  double inf_norm = 0.0;
  size_t nan_count = 0;
  for (float v : hRef) {
    if (std::isnan(v)) nan_count++;
    else inf_norm = std::fmax(inf_norm, std::fabs((double)v));
  }

  double err_cublas = 0.0, err_seq = 0.0;
  long long cublas_closer = 0, seq_closer = 0, tied = 0;
#pragma omp parallel for schedule(static) reduction(max : err_cublas, err_seq) \
    reduction(+ : cublas_closer, seq_closer, tied)
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < N; ++j) {
      double truth = 0.0;
      for (int k = 0; k < N; ++k)
        truth += (double)hA[(size_t)i * N + k] * (double)hB[(size_t)k * N + j];
      const double ec = std::fabs((double)hRef[(size_t)i * N + j] - truth);
      const double es = std::fabs((double)hSeq[(size_t)i * N + j] - truth);
      err_cublas = std::fmax(err_cublas, ec);
      err_seq = std::fmax(err_seq, es);
      if (ec < es) cublas_closer++;
      else if (es < ec) seq_closer++;
      else tied++;
    }
  }

  const long long checked = (long long)rows * N;
  printf("N=%d, checked %d rows (%lld entries) against a float64 host reference\n",
         N, rows, checked);
  printf("  cuBLAS output       : inf-norm %.4f, %zu NaNs\n", inf_norm, nan_count);
  printf("  max |cuBLAS - truth|: %.6e\n", err_cublas);
  printf("  max |seq-k  - truth|: %.6e\n", err_seq);
  printf("  closer to truth     : cuBLAS %lld, seq-k %lld, bit-identical %lld\n",
         cublas_closer, seq_closer, tied);
  if (tied == checked)
    printf("  -> cuBLAS is accumulating k sequentially here (no split-k)\n");
  else
    printf("  -> cuBLAS is using a different summation order (split-k)\n");

  BK(cublasDestroy(h));
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dSeq)); CK(cudaFree(dRef));
  return 0;
}