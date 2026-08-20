#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// K1 - global memory coalescing
//
// Same arithmetic as K0, same amount of work, one change: the mapping from
// thread id to output element. The block is now 1D (1024 threads) and is
// unflattened so that consecutive threadIdx.x land on consecutive COLUMNS:
//
//   cRow = blockIdx.y * 32 + threadIdx.x / 32     -- constant across a warp
//   cCol = blockIdx.x * 32 + threadIdx.x % 32     -- 0..31 across a warp
//
// A warp now reads 32 contiguous floats of B (128 B - one transaction instead
// of 32) and writes 32 contiguous floats of C, while its read of A collapses to
// a single address that the hardware broadcasts.
//
// The 2D-block form of this exists too (swap which of threadIdx.x/.y feeds the
// column), but the flattened 1D form makes the warp-to-address relationship
// explicit, which is the whole point of the rung.
// ---------------------------------------------------------------------------
namespace k1 {

constexpr int BLOCKSIZE = 32;

__global__ void sgemm_coalesced(int N, const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C) {
  const unsigned int cRow = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const unsigned int cCol = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < (unsigned int)N && cCol < (unsigned int)N) {
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += A[cRow * N + k] * B[k * N + cCol];
    C[cRow * N + cCol] = acc;
  }
}

inline void launch(int N, const float *A, const float *B, float *C) {
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  sgemm_coalesced<<<grid, block>>>(N, A, B, C);
}

}  // namespace k1
