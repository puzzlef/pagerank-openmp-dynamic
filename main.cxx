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
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
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
  double batchFraction = 1e-4;
  auto   fl = [](auto u) { return true; };
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog  = [&](const auto& ans, const auto& ref, const char *technique, double deletionsf, double insertionsf, int batchIndex) {
    auto err = l1NormDeltaOmp(ans.ranks, ref.ranks);
    printf(
      "{-%.3e/+%.3e batchf, %03d batchi} -> {%09.1fms, %09.1fms init, %09.1fms mark, %09.1fms comp, %03d iter, %.2e err} %s\n",
      deletionsf, insertionsf, batchIndex,
      ans.time, ans.initializationTime, ans.markingTime, ans.computationTime, ans.iterations, err, technique
    );
  };
  V tolerance = 1e-10;
  V frontierTolerance = 1e-15;
  V pruneTolerance    = 1e-15;
  // Get ranks of vertices on original graph (static).
  auto r0 = pagerankStaticOmp(xt, PagerankOptions<V>(1, 1e-100));
  // Get ranks of vertices on updated graph (dynamic).
  for (int batchIndex=0; batchIndex<BATCH_LENGTH; ++batchIndex) {
    auto y = duplicate(x);
    readTemporalOmpW(y, fstream, false, false, rows, size_t(batchFraction * size));
    y  = addSelfLoopsOmp(y, None(), fl);
    yt = transposeWithDegreeOmp(y);
    auto a0 = pagerankStaticOmp<false>(yt, PagerankOptions<V>(repeat, tolerance, frontierTolerance));
    glog(a0, r0, "pagerankStaticOmp", 0.0, 0.0, batchIndex);
    auto a1 = pagerankNaiveDynamicOmp<true>(yt, &r0.ranks, {repeat, tolerance, frontierTolerance});
    glog(a1, r0, "pagerankNaiveDynamicOmp", 0.0, 0.0, batchIndex);
    auto a2 = pagerankDynamicFrontierOmp<true, true>(x, xt, y, yt, None(), None(), &r0.ranks, {repeat, tolerance, frontierTolerance});
    glog(a2, r0, "pagerankDynamicFrontierOmp", 0.0, 0.0, batchIndex);
    auto a3 = pagerankDynamicTraversalOmp<true>(x, xt, y, yt, None(), None(), &r0.ranks, {repeat, tolerance, frontierTolerance});
    glog(a3, r0, "pagerankDynamicTraversalOmp", 0.0, 0.0, batchIndex);
    x  = move(y);
    xt = move(yt);
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
  readTemporalOmpW(x, fstream, false, false, rows, size_t(0.80 * size)); LOG(""); println(x); printf(" (80%)\n");
  auto fl = [](auto u) { return true; };
  x = addSelfLoopsOmp(x, None(), fl);  LOG(""); print(x);  printf(" (selfLoopAllVertices)\n");
  auto xt = transposeWithDegreeOmp(x); LOG(""); print(xt); printf(" (transposeWithDegree)\n");
  runExperiment(x, xt, fstream, rows, size);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
