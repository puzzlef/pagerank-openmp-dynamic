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
#define REPEAT_BATCH 1
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
  auto r0   = pagerankStaticOmp(xt, init, {1, 1e-100});
  // Get ranks of vertices on updated graph (dynamic).
  runBatches(x, rnd, [&](const auto& y, const auto& yt, double deletionsf, const auto& deletions, double insertionsf, const auto& insertions) {
    runThreads([&](int numThreads) {
      // Follow a specific result logging format, which can be easily parsed later.
      auto flog  = [&](const auto& ans, const auto& ref, const char *technique, V frontierTolerance, V pruneTolerance) {
        auto err = liNormDeltaOmp(ans.ranks, ref.ranks);
        printf(
          "{-%.3e/+%.3e batchf, %03d threads, %.0e frontier, %.0e prune} -> {%09.1fms, %03d iter, %.2e err} %s\n",
          deletionsf, insertionsf, numThreads, frontierTolerance, pruneTolerance,
          ans.time, ans.iterations, err, technique
        );
      };
      V tolerance = 1e-10;
      auto s0 = pagerankStaticOmp(yt, init, {1, 1e-100});
      // Find multi-threaded OpenMP-based Static PageRank.
      for (V frontierTolerance=1e-13; frontierTolerance>=1e-15; frontierTolerance/=10) {
        auto a0 = pagerankStaticOmp(yt, init, {repeat, tolerance, frontierTolerance});
        flog(a0, s0, "pagerankStaticOmp", frontierTolerance, 0);
      }
      for (V frontierTolerance=1e-13; frontierTolerance>=1e-15; frontierTolerance/=10) {
        for (V pruneTolerance=1e-10; pruneTolerance>=1e-20; pruneTolerance/=10) {
          auto c0 = pagerankPruneStaticOmp(y, yt, init, {repeat, tolerance, frontierTolerance, pruneTolerance});
          flog(c0, s0, "pagerankPruneStaticOmp", frontierTolerance, pruneTolerance);
        }
      }
      // Find multi-threaded OpenMP-based Naive-dynamic PageRank.
      for (V frontierTolerance=1e-13; frontierTolerance>=1e-15; frontierTolerance/=10) {
        auto a1 = pagerankStaticOmp(yt, &r0.ranks, {repeat, tolerance, frontierTolerance});
        flog(a1, s0, "pagerankNaiveDynamicOmp", frontierTolerance, 0);
      }
      for (V frontierTolerance=1e-13; frontierTolerance>=1e-15; frontierTolerance/=10) {
        for (V pruneTolerance=1e-10; pruneTolerance>=1e-20; pruneTolerance/=10) {
          auto c1 = pagerankPruneStaticOmp(y, yt, &r0.ranks, {repeat, tolerance, frontierTolerance, pruneTolerance});
          flog(c1, s0, "pagerankPruneNaiveDynamicOmp", frontierTolerance, pruneTolerance);
        }
      }
      // Find multi-threaded OpenMP-based Frontier-based Dynamic PageRank.
      for (V frontierTolerance=1e-13; frontierTolerance>=1e-15; frontierTolerance/=10) {
        auto a2 = pagerankDynamicFrontierOmp(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat, tolerance, frontierTolerance});
        flog(a2, s0, "pagerankDynamicFrontierOmp", frontierTolerance, 0);
      }
      for (V frontierTolerance=1e-13; frontierTolerance>=1e-15; frontierTolerance/=10) {
        for (V pruneTolerance=1e-10; pruneTolerance>=1e-20; pruneTolerance/=10) {
          auto c2 = pagerankPruneDynamicFrontierOmp(x, xt, y, yt, deletions, insertions, &r0.ranks, {repeat, tolerance, frontierTolerance, pruneTolerance});
          flog(c2, s0, "pagerankPruneDynamicFrontierOmp", frontierTolerance, pruneTolerance);
        }
      }
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
