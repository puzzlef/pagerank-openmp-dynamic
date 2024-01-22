#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <iostream>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of PageRank values. */
#define TYPE double
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 1
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 1
#endif
#pragma endregion




#pragma region METHODS
#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x input graph
 * @param xt transpose of input graph
 * @param fstream input file stream
 * @param rows number of rows/vetices in the graph
 * @param size number of lines/edges (temporal) in the graph
 */
template <class G, class H>
void runExperiment(G& x, H& xt, istream& fstream, size_t rows, size_t size) {
  using K = typename G::key_type;
  using V = TYPE;
  int    repeat = REPEAT_METHOD;
  double batchFraction = 1e-5;
  auto   fl = [](auto u) { return true; };
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog  = [&](const auto& ans, const auto& ref, const char *technique, double deletionsf, double insertionsf, int batchIndex, float frontierTolerance=0, float pruneTolerance=0) {
    auto err = l1NormDeltaOmp(ans.ranks, ref.ranks);
    printf(
      "{-%.3e/+%.3e batchf, %03d batchi, %.0e frontier, %.0e prune} -> {%09.1fms, %09.1fms init, %09.1fms mark, %09.1fms comp, %03d iter, %.2e err} %s\n",
      deletionsf, insertionsf, batchIndex, frontierTolerance, pruneTolerance,
      ans.time, ans.initializationTime, ans.markingTime, ans.computationTime, ans.iterations, err, technique
    );
  };
  V tolerance = 1e-10;
  V frontierTolerance = 1e-15;
  V pruneTolerance    = 1e-15;
  vector<tuple<K, K>> deletions;
  vector<tuple<K, K>> insertions;
  // Get ranks of vertices on original graph (static).
  auto r0 = pagerankStaticOmp(xt, PagerankOptions<V>(1, 1e-100));
  auto R10 = r0.ranks;
  auto R20 = r0.ranks;
  auto R550 = r0.ranks;
  auto R551 = r0.ranks;
  auto R552 = r0.ranks;
  auto R553 = r0.ranks;
  auto R554 = r0.ranks;
  auto R555 = r0.ranks;
  auto R556 = r0.ranks;
  auto R557 = r0.ranks;
  auto R558 = r0.ranks;
  auto R559 = r0.ranks;
  // Get ranks of vertices on updated graph (dynamic).
  for (int batchIndex=0; batchIndex<BATCH_LENGTH; ++batchIndex) {
    auto y = duplicate(x);
    insertions.clear();
    auto fb = [&](auto u, auto v, auto w) {
      insertions.push_back({u, v});
      y.addEdge(u, v);
    };
    readTemporalDo(fstream, false, false, rows, size_t(batchFraction * size), fb);
    updateOmpU(y);
    auto yt = transposeWithDegreeOmp(y);
    LOG(""); print(y); printf(" (insertions=%zu)\n", insertions.size());
    auto s0 = pagerankStaticOmp(yt, PagerankOptions<V>(1, 1e-100));
    auto a0 = pagerankStaticOmp<false>(yt, PagerankOptions<V>(repeat, tolerance, frontierTolerance));
    glog(a0, s0, "pagerankStaticOmp", 0.0, batchFraction, batchIndex);
    auto a1 = pagerankNaiveDynamicOmp<true>(yt, &R10, {repeat, tolerance, frontierTolerance});
    glog(a1, s0, "pagerankNaiveDynamicOmp", 0.0, batchFraction, batchIndex);
    auto a2 = pagerankDynamicTraversalOmp<true>(x, xt, y, yt, deletions, insertions, &R20, {repeat, tolerance, frontierTolerance});
    glog(a2, s0, "pagerankDynamicTraversalOmp", 0.0, batchFraction, batchIndex);
    copyValuesOmpW(R10, a1.ranks);
    copyValuesOmpW(R20, a2.ranks);
    {
      auto c550 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R550, {repeat, tolerance, 1e-6});
      glog(c550, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-6);
      auto c551 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R551, {repeat, tolerance, 1e-6, 1e-6});
      glog(c551, s0, "pagerankPruneDynamicFrontierOmp31", 0.0, batchFraction, batchIndex, 1e-6, 1e-6);
      auto c552 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R552, {repeat, tolerance, 1e-6, 1e-7});
      glog(c552, s0, "pagerankPruneDynamicFrontierOmp32", 0.0, batchFraction, batchIndex, 1e-6, 1e-7);
      auto c553 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R553, {repeat, tolerance, 1e-6, 1e-8});
      glog(c553, s0, "pagerankPruneDynamicFrontierOmp33", 0.0, batchFraction, batchIndex, 1e-6, 1e-8);
      auto c554 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R554, {repeat, tolerance, 1e-6, 1e-9});
      glog(c554, s0, "pagerankPruneDynamicFrontierOmp34", 0.0, batchFraction, batchIndex, 1e-6, 1e-9);
      auto c555 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R555, {repeat, tolerance, 1e-6, 1e-10});
      glog(c555, s0, "pagerankPruneDynamicFrontierOmp35", 0.0, batchFraction, batchIndex, 1e-6, 1e-10);
      auto c556 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R555, {repeat, tolerance, 1e-6, 1e-11});
      glog(c556, s0, "pagerankPruneDynamicFrontierOmp36", 0.0, batchFraction, batchIndex, 1e-6, 1e-11);
      auto c557 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R555, {repeat, tolerance, 1e-6, 1e-12});
      glog(c557, s0, "pagerankPruneDynamicFrontierOmp37", 0.0, batchFraction, batchIndex, 1e-6, 1e-12);
      auto c558 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R555, {repeat, tolerance, 1e-6, 1e-13});
      glog(c558, s0, "pagerankPruneDynamicFrontierOmp38", 0.0, batchFraction, batchIndex, 1e-6, 1e-13);
      auto c559 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R555, {repeat, tolerance, 1e-6, 1e-14});
      glog(c559, s0, "pagerankPruneDynamicFrontierOmp39", 0.0, batchFraction, batchIndex, 1e-6, 1e-14);
      copyValuesOmpW(R550, c550.ranks);
      copyValuesOmpW(R551, c551.ranks);
      copyValuesOmpW(R552, c552.ranks);
      copyValuesOmpW(R553, c553.ranks);
      copyValuesOmpW(R554, c554.ranks);
      copyValuesOmpW(R555, c555.ranks);
      copyValuesOmpW(R556, c556.ranks);
      copyValuesOmpW(R557, c557.ranks);
      copyValuesOmpW(R558, c558.ranks);
      copyValuesOmpW(R559, c559.ranks);
    }
    swap(x,  y);
    swap(xt, yt);
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  char *file  = argv[1];
  size_t rows = strtoull(argv[2], nullptr, 10);
  size_t size = strtoull(argv[3], nullptr, 10);
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<uint32_t> x;
  ifstream fstream(file);
  readTemporalOmpW(x, fstream, false, false, rows, size_t(0.90 * size)); LOG(""); print(x); printf(" (90%%)\n");
  auto fl = [](auto u) { return true; };
  x = addSelfLoopsOmp(x, None(), fl);  LOG(""); print(x);  printf(" (selfLoopAllVertices)\n");
  auto xt = transposeWithDegreeOmp(x); LOG(""); print(xt); printf(" (transposeWithDegree)\n");
  runExperiment(x, xt, fstream, rows, size);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
