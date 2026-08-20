#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// K0 - naive
//
// One thread per output element; each thread walks the full k dimension.
// Note the index mapping: threadIdx.x selects the ROW. That makes the 32
// threads of a warp read 32 *different rows* of A and write 32 different rows
// of C, so their addresses are N floats apart and every lane needs its own
// memory transaction. That is deliberate - it is the baseline K1 fixes.
// ---------------------------------------------------------------------------
namespace k0 {

constexpr int BLOCKSIZE = 32;

__global__ void sgemm_naive(int N, const float *__restrict__ A,
                            const float *__restrict__ B, float *__restrict__ C) {
  const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < (unsigned int)N && col < (unsigned int)N) {
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

inline void launch(int N, const float *A, const float *B, float *C) {
  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  sgemm_naive<<<grid, block>>>(N, A, B, C);
}

}  // namespace k0
