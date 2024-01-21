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
  auto glog  = [&](const auto& ans, const auto& ref, const char *technique, double deletionsf, double insertionsf, int batchIndex, float frontierTolerance=1e-12, float pruneTolerance=1e-15) {
    auto err = l1NormDeltaOmp(ans.ranks, ref.ranks);
    printf(
      "{-%.3e/+%.3e batchf, %03d batchi, %.0e frontier} -> {%09.1fms, %09.1fms init, %09.1fms mark, %09.1fms comp, %03d iter, %.2e err} %s\n",
      deletionsf, insertionsf, batchIndex, frontierTolerance,
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
  auto R30 = r0.ranks;
  auto R31 = r0.ranks;
  auto R32 = r0.ranks;
  auto R33 = r0.ranks;
  auto R34 = r0.ranks;
  auto R35 = r0.ranks;
  auto R40 = r0.ranks;
  auto R41 = r0.ranks;
  auto R42 = r0.ranks;
  auto R43 = r0.ranks;
  auto R44 = r0.ranks;
  auto R45 = r0.ranks;
  auto R50 = r0.ranks;
  auto R51 = r0.ranks;
  auto R52 = r0.ranks;
  auto R53 = r0.ranks;
  auto R54 = r0.ranks;
  auto R55 = r0.ranks;
  auto R60 = r0.ranks;
  auto R61 = r0.ranks;
  auto R62 = r0.ranks;
  auto R63 = r0.ranks;
  auto R64 = r0.ranks;
  auto R65 = r0.ranks;
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
    auto a30 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R30, {repeat, tolerance, 1e-10});
    glog(a30, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-10);
    auto a31 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R31, {repeat, tolerance, 1e-11});
    glog(a31, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-12);
    auto a32 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R32, {repeat, tolerance, 1e-12});
    glog(a32, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-14);
    auto a33 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R33, {repeat, tolerance, 1e-13});
    glog(a33, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-16);
    auto a34 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R34, {repeat, tolerance, 1e-14});
    glog(a34, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-18);
    auto a35 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R35, {repeat, tolerance, 1e-15});
    glog(a35, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-20);
    auto b40 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R40, {repeat, tolerance, 1e-10});
    glog(b40, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-10);
    auto b41 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R41, {repeat, tolerance, 1e-11});
    glog(b41, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-12);
    auto b42 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R42, {repeat, tolerance, 1e-12});
    glog(b42, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-14);
    auto b43 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R43, {repeat, tolerance, 1e-13});
    glog(b43, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-16);
    auto b44 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R44, {repeat, tolerance, 1e-14});
    glog(b44, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-18);
    auto b45 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R45, {repeat, tolerance, 1e-15});
    glog(b45, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-20);
    auto c50 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R50, {repeat, tolerance, 1e-1});
    glog(c50, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-1);
    auto c51 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R51, {repeat, tolerance, 1e-2});
    glog(c51, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-2);
    auto c52 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R52, {repeat, tolerance, 1e-3});
    glog(c52, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-3);
    auto c53 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R53, {repeat, tolerance, 1e-4});
    glog(c53, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-4);
    auto c54 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R54, {repeat, tolerance, 1e-5});
    glog(c54, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-5);
    auto c55 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R55, {repeat, tolerance, 1e-6});
    glog(c55, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-6);
    auto d60 = pagerankDynamicFrontierOmp<true, true, true, 4>(x, xt, y, yt, deletions, insertions, &R60, {repeat, tolerance, 1e-1});
    glog(d60, s0, "pagerankDynamicFrontierOmp4", 0.0, batchFraction, batchIndex, 1e-1);
    auto d61 = pagerankDynamicFrontierOmp<true, true, true, 4>(x, xt, y, yt, deletions, insertions, &R61, {repeat, tolerance, 1e-2});
    glog(d61, s0, "pagerankDynamicFrontierOmp4", 0.0, batchFraction, batchIndex, 1e-2);
    auto d62 = pagerankDynamicFrontierOmp<true, true, true, 4>(x, xt, y, yt, deletions, insertions, &R62, {repeat, tolerance, 1e-3});
    glog(d62, s0, "pagerankDynamicFrontierOmp4", 0.0, batchFraction, batchIndex, 1e-3);
    auto d63 = pagerankDynamicFrontierOmp<true, true, true, 4>(x, xt, y, yt, deletions, insertions, &R63, {repeat, tolerance, 1e-4});
    glog(d63, s0, "pagerankDynamicFrontierOmp4", 0.0, batchFraction, batchIndex, 1e-4);
    auto d64 = pagerankDynamicFrontierOmp<true, true, true, 4>(x, xt, y, yt, deletions, insertions, &R64, {repeat, tolerance, 1e-5});
    glog(d64, s0, "pagerankDynamicFrontierOmp4", 0.0, batchFraction, batchIndex, 1e-5);
    auto d65 = pagerankDynamicFrontierOmp<true, true, true, 4>(x, xt, y, yt, deletions, insertions, &R65, {repeat, tolerance, 1e-6});
    glog(d65, s0, "pagerankDynamicFrontierOmp4", 0.0, batchFraction, batchIndex, 1e-6);
    copyValuesOmpW(R10, a1.ranks);
    copyValuesOmpW(R20, a2.ranks);
    copyValuesOmpW(R30, a30.ranks);
    copyValuesOmpW(R31, a31.ranks);
    copyValuesOmpW(R32, a32.ranks);
    copyValuesOmpW(R33, a33.ranks);
    copyValuesOmpW(R34, a34.ranks);
    copyValuesOmpW(R35, a35.ranks);
    copyValuesOmpW(R40, b40.ranks);
    copyValuesOmpW(R41, b41.ranks);
    copyValuesOmpW(R42, b42.ranks);
    copyValuesOmpW(R43, b43.ranks);
    copyValuesOmpW(R44, b44.ranks);
    copyValuesOmpW(R45, b45.ranks);
    copyValuesOmpW(R50, c50.ranks);
    copyValuesOmpW(R51, c51.ranks);
    copyValuesOmpW(R52, c52.ranks);
    copyValuesOmpW(R53, c53.ranks);
    copyValuesOmpW(R54, c54.ranks);
    copyValuesOmpW(R55, c55.ranks);
    copyValuesOmpW(R60, d60.ranks);
    copyValuesOmpW(R61, d61.ranks);
    copyValuesOmpW(R62, d62.ranks);
    copyValuesOmpW(R63, d63.ranks);
    copyValuesOmpW(R64, d64.ranks);
    copyValuesOmpW(R65, d65.ranks);
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
