// Records the hardware numbers the blocktiling parameters are chosen against.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count == 0) {
    printf("no CUDA devices found\n");
    return 1;
  }

  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);

  for (int dev = 0; dev < count; ++dev) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    int sm_clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);

    // Ada/Ampere-class boards report bandwidth as clock (kHz) * bus (bits) * 2 (DDR).
    const double bw_gbs =
        2.0 * mem_clock_khz * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;

    printf("Device %d: %s\n", dev, p.name);
    printf("  Compute capability            : %d.%d  (-arch=sm_%d%d)\n",
           p.major, p.minor, p.major, p.minor);
    printf("  CUDA driver / runtime version : %d.%d / %d.%d\n", driver / 1000,
           (driver % 100) / 10, runtime / 1000, (runtime % 100) / 10);
    printf("  SM count                      : %d\n", p.multiProcessorCount);
    printf("  Global memory                 : %.0f MiB\n",
           p.totalGlobalMem / 1048576.0);
    printf("  L2 cache size                 : %d KiB\n", p.l2CacheSize / 1024);
    printf("  Memory bus width              : %d bit\n", p.memoryBusWidth);
    printf("  Memory clock                  : %.0f MHz\n", mem_clock_khz / 1000.0);
    printf("  Theoretical bandwidth         : %.1f GB/s\n", bw_gbs);
    printf("  SM clock (boost)              : %.0f MHz\n", sm_clock_khz / 1000.0);
    printf("  Shared memory per block       : %zu B\n", p.sharedMemPerBlock);
    printf("  Shared memory per block (optin): %zu B\n", p.sharedMemPerBlockOptin);
    printf("  Shared memory per SM          : %zu B\n", p.sharedMemPerMultiprocessor);
    printf("  Registers per block           : %d\n", p.regsPerBlock);
    printf("  Registers per SM              : %d\n", p.regsPerMultiprocessor);
    printf("  Max threads per block         : %d\n", p.maxThreadsPerBlock);
    printf("  Max threads per SM            : %d\n", p.maxThreadsPerMultiProcessor);
    printf("  Warp size                     : %d\n", p.warpSize);
    printf("  Max block dim                 : (%d, %d, %d)\n", p.maxThreadsDim[0],
           p.maxThreadsDim[1], p.maxThreadsDim[2]);
    printf("  Max grid dim                  : (%d, %d, %d)\n", p.maxGridSize[0],
           p.maxGridSize[1], p.maxGridSize[2]);
    printf("  Concurrent kernels            : %d\n", p.concurrentKernels);
    printf("  Async engine count            : %d\n", p.asyncEngineCount);
    printf("  Unified addressing            : %d\n", p.unifiedAddressing);
  }
  return 0;
}
