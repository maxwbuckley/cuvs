// Minimal: just cu_roaring + CUDA, no cuvs/raft headers
#include <cuda_runtime.h>
#include <cu_roaring/types.cuh>
#include <cu_roaring/detail/upload_ids.cuh>
#include <cu_roaring/detail/decompress.cuh>
#include <cu_roaring/device/make_view.cuh>
#include <cstdio>
#include <vector>
#include <random>

namespace cu_roaring { void gpu_roaring_free(GpuRoaring& bitmap); }

int main() {
  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<uint32_t> ids;
  for (int i = 0; i < 1000000; ++i)
    if (dist(gen) < 0.50) ids.push_back(static_cast<uint32_t>(i));
  fprintf(stderr, "Generated %zu IDs\n", ids.size());

  auto bm = cu_roaring::upload_from_ids(ids.data(), (uint32_t)ids.size(), 1000000u);
  cudaDeviceSynchronize();
  fprintf(stderr, "n_containers=%u bmp=%u arr=%u negated=%d total_card=%lu\n",
          bm.n_containers, bm.n_bitmap_containers, bm.n_array_containers,
          bm.negated, bm.total_cardinality);

  // Verify by decompressing
  uint32_t* d_bits = cu_roaring::decompress_to_bitset(bm);
  uint32_t n_words = (1000000 + 31) / 32;
  std::vector<uint32_t> h_bits(n_words);
  cudaMemcpy(h_bits.data(), d_bits, n_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_bits);

  int count = 0;
  for (uint32_t w : h_bits) count += __builtin_popcount(w);
  fprintf(stderr, "Decompressed popcount: %d (expected ~%zu)\n", count, ids.size());

  cu_roaring::gpu_roaring_free(bm);
  fprintf(stderr, "Done\n");
  return 0;
}
