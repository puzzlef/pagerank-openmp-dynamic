#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
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
#pragma region EXPERIMENTAL SETUP
/**
 * Run a function on each batch update, with a specified range of batch sizes.
 * @param x original graph
 * @param rnd random number generator
 * @param fn function to run on each batch update
 */
template <class G, class R, class F>
inline void runBatches(const G& x, R& rnd, F fn) {
  auto fl = [](auto u) { return true; };
  double d = BATCH_DELETIONS_BEGIN;
  double i = BATCH_INSERTIONS_BEGIN;
  for (int epoch=0;; ++epoch) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      for (int sequence=0; sequence<BATCH_LENGTH; ++sequence) {
        auto deletions  = generateEdgeDeletions (rnd, y, size_t(d * x.size()/2), 1, x.span()-1, true);
        auto insertions = generateEdgeInsertions(rnd, y, size_t(i * x.size()/2), 1, x.span()-1, true, None());
        tidyBatchUpdateU(deletions, insertions, y);
        applyBatchUpdateOmpU(y, deletions, insertions);
        addSelfLoopsOmpU(y, None(), fl);
        auto yt = transposeWithDegreeOmp(y);
        fn(y, yt, d, deletions, i, insertions, sequence, epoch);
      }
    }
    if (d>=BATCH_DELETIONS_END && i>=BATCH_INSERTIONS_END) break;
    d BATCH_DELETIONS_STEP;
    i BATCH_INSERTIONS_STEP;
    d = min(d, double(BATCH_DELETIONS_END));
    i = min(i, double(BATCH_INSERTIONS_END));
  }
}


/**
 * Run a function on each number of threads, for a specific epoch.
 * @param epoch epoch number
 * @param fn function to run on each number of threads
 */
template <class F>
inline void runThreadsWithBatch(int epoch, F fn) {
  int t = NUM_THREADS_BEGIN;
  for (int l=0; l<epoch && t<=NUM_THREADS_END; ++l)
    t NUM_THREADS_STEP;
  omp_set_num_threads(t);
  fn(t);
  omp_set_num_threads(MAX_THREADS);
}


/**
 * Run a function on each number of threads, with a specified range of thread counts.
 * @param fn function to run on each number of threads
 */
template <class F>
inline void runThreadsAll(F fn) {
  for (int t=NUM_THREADS_BEGIN; t<=NUM_THREADS_END; t NUM_THREADS_STEP) {
    omp_set_num_threads(t);
    fn(t);
    omp_set_num_threads(MAX_THREADS);
  }
}


/**
 * Run a function on each number of threads, with a specified range of thread counts or for a specific epoch (depending on NUM_THREADS_MODE).
 * @param epoch epoch number
 * @param fn function to run on each number of threads
 */
template <class F>
inline void runThreads(int epoch, F fn) {
  if (NUM_THREADS_MODE=="with-batch") runThreadsWithBatch(epoch, fn);
  else runThreadsAll(fn);
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x original graph
 * @param xt transposed graph
 */
template <class G, class H>
void runExperiment(const G& x, const H& xt) {
  using  K = typename G::key_type;
  using  V = TYPE;
  random_device dev;
  default_random_engine rnd(dev());
  int repeat = REPEAT_METHOD;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog  = [&](const auto& ans, const auto& ref, const char *technique, int numThreads, double deletionsf, double insertionsf, V frontierTolerance, V pruneTolerance) {
    auto err = l1NormDeltaOmp(ans.ranks, ref.ranks);
    printf(
      "{-%.3e/+%.3e batchf, %03d threads} -> {%09.1fms, %09.1fms init, %09.1fms mark, %09.1fms comp, %03d iter, %.2e err, %.3e aff} %s\n",
      deletionsf, insertionsf, numThreads,
      ans.time, ans.initializationTime, ans.markingTime, ans.computationTime, ans.iterations, err, double(ans.affectedVertices), technique
    );
  };
  // Get ranks of vertices on original graph (static).
  auto r0 = pagerankStaticOmp(xt, PagerankOptions<V>(1, 1e-100));
  // Get ranks of vertices on updated graph (dynamic).
  runBatches(x, rnd, [&](const auto& y, const auto& yt, double deletionsf, const auto& deletions, double insertionsf, const auto& insertions, int sequence, int epoch) {
    runThreads(epoch, [&](int numThreads) {
      auto flog = [&](const auto& ans, const auto& ref, const char *technique, V frontierTolerance, V pruneTolerance) {
        glog(ans, ref, technique, numThreads, deletionsf, insertionsf, frontierTolerance, pruneTolerance);
      };
      V tolerance = 1e-10;
      V frontierTolerance = 1e-15;
      // Find multi-threaded OpenMP-based Frontier-based Dynamic PageRank.
      auto a2 = pagerankDynamicFrontierOmp<true, true>(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat, tolerance, frontierTolerance});
      flog(a2, a2, "pagerankDynamicFrontierOmp", frontierTolerance, 0.0);
      // Find multi-threaded OpenMP-based Traversal-based Dynamic PageRank.
      auto a3 = pagerankDynamicTraversalOmp<true>(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat, tolerance, frontierTolerance});
      flog(a3, a3, "pagerankDynamicTraversalOmp", frontierTolerance, 0.0);
    });
  });
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  char *file = argv[1];
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<uint32_t> x;
  readMtxOmpW(x, file); LOG(""); println(x);
  auto fl = [](auto u) { return true; };
  x = addSelfLoopsOmp(x, None(), fl);  LOG(""); print(x);  printf(" (selfLoopAllVertices)\n");
  auto xt = transposeWithDegreeOmp(x); LOG(""); print(xt); printf(" (transposeWithDegree)\n");
  runExperiment(x, xt);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
