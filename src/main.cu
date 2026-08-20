// CLI driver: pick a kernel id and a size, check correctness against cuBLAS,
// then benchmark. See --help for the flags.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "runner.h"
#include "utils.h"

namespace {

struct Options {
  int kernel = KERNEL_CUBLAS;
  int N = 1024;
  int warmup = 5;
  int iters = 50;
  unsigned seed = 1234u;
  double tol = 1e-3;
  bool verify = true;
  bool csv = false;
  bool csv_header = false;
  bool cpu_check = false;
};

void usage(const char *prog) {
  printf(
      "usage: %s --kernel <id|cublas> --size <N> [options]\n"
      "       %s <id|cublas> <N> [options]\n"
      "\n"
      "kernels:\n"
      "  cublas   cuBLAS SGEMM baseline (FP32, no TF32)\n"
      "\n"
      "options:\n"
      "  --warmup <n>   discarded iterations before timing (default 5)\n"
      "  --iters <n>    timed iterations, median reported (default 50)\n"
      "  --seed <n>     RNG seed for input generation (default 1234)\n"
      "  --tol <x>      max relative error accepted (default 1e-3)\n"
      "  --no-verify    skip the cuBLAS correctness check\n"
      "  --csv          emit one CSV row instead of the human-readable report\n"
      "  --csv-header   print the CSV header line and exit\n"
      "  --cpu-check    also validate cuBLAS itself against a CPU reference\n"
      "                 (O(N^3) on the host -- keep N small, e.g. 128)\n",
      prog, prog);
}

bool parse_args(int argc, char **argv, Options &o) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s requires a value\n", flag);
        exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    } else if (a == "--kernel") {
      o.kernel = parse_kernel_id(need("--kernel"));
    } else if (a == "--size" || a == "-n") {
      o.N = std::stoi(need("--size"));
    } else if (a == "--warmup") {
      o.warmup = std::stoi(need("--warmup"));
    } else if (a == "--iters") {
      o.iters = std::stoi(need("--iters"));
    } else if (a == "--seed") {
      o.seed = static_cast<unsigned>(std::stoul(need("--seed")));
    } else if (a == "--tol") {
      o.tol = std::stod(need("--tol"));
    } else if (a == "--no-verify") {
      o.verify = false;
    } else if (a == "--csv") {
      o.csv = true;
    } else if (a == "--csv-header") {
      o.csv_header = true;
    } else if (a == "--cpu-check") {
      o.cpu_check = true;
    } else if (!a.empty() && a[0] == '-') {
      fprintf(stderr, "unknown flag: %s\n", a.c_str());
      return false;
    } else if (positional == 0) {
      o.kernel = parse_kernel_id(a);
      positional++;
    } else if (positional == 1) {
      o.N = std::stoi(a);
      positional++;
    } else {
      fprintf(stderr, "unexpected argument: %s\n", a.c_str());
      return false;
    }
  }
  if (o.kernel == INT_MIN) {
    fprintf(stderr, "unknown kernel (no ladder kernels yet -- expected 'cublas')\n");
    return false;
  }
  if (o.N <= 0 || o.iters <= 0 || o.warmup < 0) {
    fprintf(stderr, "size and iters must be positive, warmup non-negative\n");
    return false;
  }
  return true;
}

const char *kCsvHeader =
    "kernel,kernel_id,N,median_ms,min_ms,p95_ms,gflops,max_rel_err,warmup,iters";

}  // namespace

int main(int argc, char **argv) {
  Options o;
  if (!parse_args(argc, argv, o)) return EXIT_FAILURE;
  if (o.csv_header) {
    printf("%s\n", kCsvHeader);
    return EXIT_SUCCESS;
  }

  const int N = o.N;
  const size_t elems = static_cast<size_t>(N) * N;
  const size_t bytes = elems * sizeof(float);

  std::vector<float> hA(elems), hB(elems), hC(elems), hRef(elems);
  fill_random(hA, o.seed);
  fill_random(hB, o.seed + 1u);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMalloc(&dRef, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle = make_cublas_handle();

  // ---- cuBLAS reference -----------------------------------------------
  CUDA_CHECK(cudaMemset(dRef, 0, bytes));
  cublas_sgemm(handle, N, dA, dB, dRef);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, bytes, cudaMemcpyDeviceToHost));

  if (o.cpu_check) {
    std::vector<float> hCpu(elems);
    cpu_sgemm_reference(N, hA, hB, hCpu);
    const ErrorReport e = compare(hRef, hCpu);
    printf("cuBLAS vs CPU reference (N=%d): max_abs=%.3e  rel=%.3e  %s\n", N,
           e.max_abs, e.rel, e.rel < o.tol ? "OK" : "FAIL");
    if (e.rel >= o.tol) {
      fprintf(stderr,
              "cuBLAS operand order is wrong -- nothing downstream is "
              "trustworthy\n");
      return EXIT_FAILURE;
    }
  }

  // ---- correctness ----------------------------------------------------
  ErrorReport err{};
  if (o.verify) {
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    run_kernel(o.kernel, N, dA, dB, dC, handle);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    err = compare(hC, hRef);
    if (!(err.rel < o.tol)) {
      fprintf(stderr,
              "[FAIL] %s N=%d: relative error %.3e >= tol %.3e "
              "(max_abs=%.3e at index %d)\n",
              kernel_name(o.kernel), N, err.rel, o.tol, err.max_abs,
              err.bad_index);
      return EXIT_FAILURE;
    }
  }

  // ---- benchmark ------------------------------------------------------
  const BenchResult b =
      benchmark(o.kernel, N, dA, dB, dC, handle, o.warmup, o.iters);
  const double gf = gflops(N, b.median_ms);

  if (o.csv) {
    printf("%s,%d,%d,%.6f,%.6f,%.6f,%.2f,%.3e,%d,%d\n", kernel_name(o.kernel),
           o.kernel, N, b.median_ms, b.min_ms, b.p95_ms, gf,
           o.verify ? err.rel : -1.0, o.warmup, b.iters);
  } else {
    printf("kernel      : %s (id %d)\n", kernel_name(o.kernel), o.kernel);
    printf("size        : N=%d  (%.1f MiB per matrix)\n", N, bytes / 1048576.0);
    printf("timing      : %d warmup + %d timed iterations, median reported\n",
           o.warmup, b.iters);
    printf("median      : %.4f ms  ->  %.2f GFLOPS\n", b.median_ms, gf);
    printf("min / p95   : %.4f ms / %.4f ms\n", b.min_ms, b.p95_ms);
    if (o.verify)
      printf("correctness : rel=%.3e (tol %.1e), max_abs=%.3e, max_elem_rel=%.3e -> PASS\n",
             err.rel, o.tol, err.max_abs, err.max_elem_rel);
    else
      printf("correctness : skipped (--no-verify)\n");
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dRef));
  return EXIT_SUCCESS;
}
