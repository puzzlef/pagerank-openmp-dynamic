#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "src/main.hxx"

using namespace std;




// Fixed config
#ifndef TYPE
#define TYPE double
#endif
#ifndef MAX_THREADS
#define MAX_THREADS 32
#endif
#ifndef REPEAT_BATCH
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
#define REPEAT_METHOD 1
#endif




// PERFORM EXPERIMENT
// ------------------

template <class G, class R, class F>
inline void runAbsoluteBatches(const G& x, R& rnd, F fn) {
  auto fl = [](auto u) { return true; };
  size_t d = BATCH_DELETIONS_BEGIN;
  size_t i = BATCH_INSERTIONS_BEGIN;
  while (true) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      auto deletions  = removeRandomEdges(y, rnd, d, 1, x.span()-1);
      auto insertions = addRandomEdges   (y, rnd, i, 1, x.span()-1, None());
      addSelfLoopsOmpU(y, None(), fl);
      auto yt = transposeWithDegreeOmp(y);
      fn(y, yt, d, deletions, i, insertions);
    }
    if (d>=BATCH_DELETIONS_END && i>=BATCH_INSERTIONS_END) break;
    d BATCH_DELETIONS_STEP;
    i BATCH_INSERTIONS_STEP;
    d = min(d, size_t(BATCH_DELETIONS_END));
    i = min(i, size_t(BATCH_INSERTIONS_END));
  }
}


template <class G, class R, class F>
inline void runRelativeBatches(const G& x, R& rnd, F fn) {
  auto fl = [](auto u) { return true; };
  double d = BATCH_DELETIONS_BEGIN;
  double i = BATCH_INSERTIONS_BEGIN;
  while (true) {
    for (int r=0; r<REPEAT_BATCH; ++r) {
      auto y  = duplicate(x);
      auto deletions  = removeRandomEdges(y, rnd, size_t(d * x.size() + 0.5), 1, x.span()-1);
      auto insertions = addRandomEdges   (y, rnd, size_t(i * x.size() + 0.5), 1, x.span()-1, None());
      addSelfLoopsOmpU(y, None(), fl);
      auto yt = transposeWithDegreeOmp(y);
      fn(y, yt, d, deletions, i, insertions);
    }
    if (d>=BATCH_DELETIONS_END && i>=BATCH_INSERTIONS_END) break;
    d BATCH_DELETIONS_STEP;
    i BATCH_INSERTIONS_STEP;
    d = min(d, double(BATCH_DELETIONS_END));
    i = min(i, double(BATCH_INSERTIONS_END));
  }
}


template <class G, class R, class F>
inline void runBatches(const G& x, R& rnd, F fn) {
  if (BATCH_UNIT=="%") runRelativeBatches(x, rnd, fn);
  else runAbsoluteBatches(x, rnd, fn);
}


template <class F>
inline void runThreads(F fn) {
  for (int t=NUM_THREADS_BEGIN; t<=NUM_THREADS_END; t NUM_THREADS_STEP) {
    omp_set_num_threads(t);
    fn(t);
    omp_set_num_threads(MAX_THREADS);
  }
}




template <class G, class H>
void runExperiment(const G& x, const H& xt) {
  using  K = typename G::key_type;
  using  V = TYPE;
  vector<V> *init = nullptr;
  random_device dev;
  default_random_engine rnd(dev());
  int repeat = REPEAT_METHOD;
  // Get ranks of vertices on original graph (static).
  auto r0   = pagerankNaiveDynamicOmp(xt, init, {1, 1e-100});
  // Get ranks of vertices on updated graph (dynamic).
  runBatches(x, rnd, [&](const auto& y, const auto& yt, double deletionsf, const auto& deletions, double insertionsf, const auto& insertions) {
    runThreads([&](int numThreads) {
        // Follow a specific result logging format, which can be easily parsed later.
        auto flog  = [&](const auto& ans, const auto& ref, const char *technique) {
          auto err = liNormDeltaOmp(ans.ranks, ref.ranks);
          printf(
            "{-%.3e/+%.3e batchf, %03d threads} -> {%09.1fms, %03d iter, %.2e err} %s\n",
            deletionsf, insertionsf, numThreads, ans.time, ans.iterations, err, technique
          );
        };
        auto s0 = pagerankNaiveDynamicOmp(yt, init, {1, 1e-100});
        // Find multi-threaded OpenMP-based Static PageRank.
        auto a0 = pagerankNaiveDynamicOmp(yt, init, {repeat});
        flog(a0, s0, "pagerankStaticOmp");
        auto b0 = pagerankPruneNaiveDynamicOmp(y, yt, init, {repeat});
        flog(b0, s0, "pagerankPruneStaticOmp");
        // Find multi-threaded OpenMP-based Naive-dynamic PageRank.
        auto a1 = pagerankNaiveDynamicOmp(yt, &r0.ranks, {repeat});
        flog(a1, s0, "pagerankNaiveDynamicOmp");
        auto b1 = pagerankPruneNaiveDynamicOmp(y, yt, &r0.ranks, {repeat});
        flog(b1, s0, "pagerankPruneNaiveDynamicOmp");
        // Find multi-threaded OpenMP-based Frontier-based Dynamic PageRank.
        auto a2 = pagerankDynamicFrontierOmp(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat});
        flog(a2, s0, "pagerankDynamicFrontierOmp");
        auto b2 = pagerankPruneDynamicFrontierOmp(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat});
        flog(b2, s0, "pagerankPruneDynamicFrontierOmp");
      });
    });
}


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
