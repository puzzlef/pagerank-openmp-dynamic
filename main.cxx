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
  auto R330 = r0.ranks;
  auto R331 = r0.ranks;
  auto R332 = r0.ranks;
  auto R333 = r0.ranks;
  auto R334 = r0.ranks;
  auto R335 = r0.ranks;
  auto R340 = r0.ranks;
  auto R341 = r0.ranks;
  auto R342 = r0.ranks;
  auto R343 = r0.ranks;
  auto R344 = r0.ranks;
  auto R345 = r0.ranks;
  auto R350 = r0.ranks;
  auto R351 = r0.ranks;
  auto R352 = r0.ranks;
  auto R353 = r0.ranks;
  auto R354 = r0.ranks;
  auto R355 = r0.ranks;
  auto R450 = r0.ranks;
  auto R451 = r0.ranks;
  auto R452 = r0.ranks;
  auto R453 = r0.ranks;
  auto R454 = r0.ranks;
  auto R455 = r0.ranks;
  auto R460 = r0.ranks;
  auto R461 = r0.ranks;
  auto R462 = r0.ranks;
  auto R463 = r0.ranks;
  auto R464 = r0.ranks;
  auto R465 = r0.ranks;
  auto R470 = r0.ranks;
  auto R471 = r0.ranks;
  auto R472 = r0.ranks;
  auto R473 = r0.ranks;
  auto R474 = r0.ranks;
  auto R475 = r0.ranks;
  auto R550 = r0.ranks;
  auto R551 = r0.ranks;
  auto R552 = r0.ranks;
  auto R553 = r0.ranks;
  auto R554 = r0.ranks;
  auto R555 = r0.ranks;
  auto R560 = r0.ranks;
  auto R561 = r0.ranks;
  auto R562 = r0.ranks;
  auto R563 = r0.ranks;
  auto R564 = r0.ranks;
  auto R565 = r0.ranks;
  auto R570 = r0.ranks;
  auto R571 = r0.ranks;
  auto R572 = r0.ranks;
  auto R573 = r0.ranks;
  auto R574 = r0.ranks;
  auto R575 = r0.ranks;
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
      auto a330 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R330, {repeat, tolerance, 1e-13});
      glog(a330, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-13);
      auto a331 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R331, {repeat, tolerance, 1e-13, 1e-13});
      glog(a331, s0, "pagerankPruneDynamicFrontierOmp11", 0.0, batchFraction, batchIndex, 1e-13, 1e-13);
      auto a332 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R332, {repeat, tolerance, 1e-13, 1e-15});
      glog(a332, s0, "pagerankPruneDynamicFrontierOmp12", 0.0, batchFraction, batchIndex, 1e-13, 1e-15);
      auto a333 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R333, {repeat, tolerance, 1e-13, 1e-17});
      glog(a333, s0, "pagerankPruneDynamicFrontierOmp13", 0.0, batchFraction, batchIndex, 1e-13, 1e-17);
      auto a334 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R334, {repeat, tolerance, 1e-13, 1e-19});
      glog(a334, s0, "pagerankPruneDynamicFrontierOmp14", 0.0, batchFraction, batchIndex, 1e-13, 1e-19);
      auto a335 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R335, {repeat, tolerance, 1e-13, 1e-21});
      glog(a335, s0, "pagerankPruneDynamicFrontierOmp15", 0.0, batchFraction, batchIndex, 1e-13, 1e-21);
      copyValuesOmpW(R330, a330.ranks);
      copyValuesOmpW(R331, a331.ranks);
      copyValuesOmpW(R332, a332.ranks);
      copyValuesOmpW(R333, a333.ranks);
      copyValuesOmpW(R334, a334.ranks);
      copyValuesOmpW(R335, a335.ranks);
    }
    {
      auto a340 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R340, {repeat, tolerance, 1e-14});
      glog(a340, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-14);
      auto a341 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R341, {repeat, tolerance, 1e-14, 1e-14});
      glog(a341, s0, "pagerankPruneDynamicFrontierOmp11", 0.0, batchFraction, batchIndex, 1e-14, 1e-14);
      auto a342 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R342, {repeat, tolerance, 1e-14, 1e-16});
      glog(a342, s0, "pagerankPruneDynamicFrontierOmp12", 0.0, batchFraction, batchIndex, 1e-14, 1e-16);
      auto a343 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R343, {repeat, tolerance, 1e-14, 1e-18});
      glog(a343, s0, "pagerankPruneDynamicFrontierOmp13", 0.0, batchFraction, batchIndex, 1e-14, 1e-18);
      auto a344 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R344, {repeat, tolerance, 1e-14, 1e-20});
      glog(a344, s0, "pagerankPruneDynamicFrontierOmp14", 0.0, batchFraction, batchIndex, 1e-14, 1e-20);
      auto a345 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R345, {repeat, tolerance, 1e-14, 1e-22});
      glog(a345, s0, "pagerankPruneDynamicFrontierOmp15", 0.0, batchFraction, batchIndex, 1e-14, 1e-22);
      copyValuesOmpW(R340, a340.ranks);
      copyValuesOmpW(R341, a341.ranks);
      copyValuesOmpW(R342, a342.ranks);
      copyValuesOmpW(R343, a343.ranks);
      copyValuesOmpW(R344, a344.ranks);
      copyValuesOmpW(R345, a345.ranks);
    }
    {
      auto a350 = pagerankDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R350, {repeat, tolerance, 1e-15});
      glog(a350, s0, "pagerankDynamicFrontierOmp1", 0.0, batchFraction, batchIndex, 1e-15);
      auto a351 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R351, {repeat, tolerance, 1e-15, 1e-15});
      glog(a351, s0, "pagerankPruneDynamicFrontierOmp11", 0.0, batchFraction, batchIndex, 1e-15, 1e-15);
      auto a352 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R352, {repeat, tolerance, 1e-15, 1e-17});
      glog(a352, s0, "pagerankPruneDynamicFrontierOmp12", 0.0, batchFraction, batchIndex, 1e-15, 1e-17);
      auto a353 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R353, {repeat, tolerance, 1e-15, 1e-19});
      glog(a353, s0, "pagerankPruneDynamicFrontierOmp13", 0.0, batchFraction, batchIndex, 1e-15, 1e-19);
      auto a354 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R354, {repeat, tolerance, 1e-15, 1e-21});
      glog(a354, s0, "pagerankPruneDynamicFrontierOmp14", 0.0, batchFraction, batchIndex, 1e-15, 1e-21);
      auto a355 = pagerankPruneDynamicFrontierOmp<true, true, true, 1>(x, xt, y, yt, deletions, insertions, &R355, {repeat, tolerance, 1e-15, 1e-23});
      glog(a355, s0, "pagerankPruneDynamicFrontierOmp15", 0.0, batchFraction, batchIndex, 1e-15, 1e-23);
      copyValuesOmpW(R350, a350.ranks);
      copyValuesOmpW(R351, a351.ranks);
      copyValuesOmpW(R352, a352.ranks);
      copyValuesOmpW(R353, a353.ranks);
      copyValuesOmpW(R354, a354.ranks);
      copyValuesOmpW(R355, a355.ranks);
    }
    {
      auto b450 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R450, {repeat, tolerance, 1e-15});
      glog(b450, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-15);
      auto b451 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R451, {repeat, tolerance, 1e-15, 1e-15});
      glog(b451, s0, "pagerankPruneDynamicFrontierOmp21", 0.0, batchFraction, batchIndex, 1e-15, 1e-15);
      auto b452 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R452, {repeat, tolerance, 1e-15, 1e-17});
      glog(b452, s0, "pagerankPruneDynamicFrontierOmp22", 0.0, batchFraction, batchIndex, 1e-15, 1e-17);
      auto b453 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R453, {repeat, tolerance, 1e-15, 1e-19});
      glog(b453, s0, "pagerankPruneDynamicFrontierOmp23", 0.0, batchFraction, batchIndex, 1e-15, 1e-19);
      auto b454 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R454, {repeat, tolerance, 1e-15, 1e-21});
      glog(b454, s0, "pagerankPruneDynamicFrontierOmp24", 0.0, batchFraction, batchIndex, 1e-15, 1e-21);
      auto b455 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R455, {repeat, tolerance, 1e-15, 1e-23});
      glog(b455, s0, "pagerankPruneDynamicFrontierOmp25", 0.0, batchFraction, batchIndex, 1e-15, 1e-23);
      copyValuesOmpW(R450, b450.ranks);
      copyValuesOmpW(R451, b451.ranks);
      copyValuesOmpW(R452, b452.ranks);
      copyValuesOmpW(R453, b453.ranks);
      copyValuesOmpW(R454, b454.ranks);
      copyValuesOmpW(R455, b455.ranks);
    }
    {
      auto b460 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R460, {repeat, tolerance, 1e-16});
      glog(b460, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-16);
      auto b461 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R461, {repeat, tolerance, 1e-16, 1e-16});
      glog(b461, s0, "pagerankPruneDynamicFrontierOmp21", 0.0, batchFraction, batchIndex, 1e-16, 1e-16);
      auto b462 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R462, {repeat, tolerance, 1e-16, 1e-18});
      glog(b462, s0, "pagerankPruneDynamicFrontierOmp22", 0.0, batchFraction, batchIndex, 1e-16, 1e-18);
      auto b463 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R463, {repeat, tolerance, 1e-16, 1e-20});
      glog(b463, s0, "pagerankPruneDynamicFrontierOmp23", 0.0, batchFraction, batchIndex, 1e-16, 1e-20);
      auto b464 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R464, {repeat, tolerance, 1e-16, 1e-22});
      glog(b464, s0, "pagerankPruneDynamicFrontierOmp24", 0.0, batchFraction, batchIndex, 1e-16, 1e-22);
      auto b465 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R465, {repeat, tolerance, 1e-16, 1e-24});
      glog(b465, s0, "pagerankPruneDynamicFrontierOmp25", 0.0, batchFraction, batchIndex, 1e-16, 1e-24);
      copyValuesOmpW(R460, b460.ranks);
      copyValuesOmpW(R461, b461.ranks);
      copyValuesOmpW(R462, b462.ranks);
      copyValuesOmpW(R463, b463.ranks);
      copyValuesOmpW(R464, b464.ranks);
      copyValuesOmpW(R465, b465.ranks);
    }
    {
      auto b470 = pagerankDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R470, {repeat, tolerance, 1e-17});
      glog(b470, s0, "pagerankDynamicFrontierOmp2", 0.0, batchFraction, batchIndex, 1e-17);
      auto b471 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R471, {repeat, tolerance, 1e-17, 1e-17});
      glog(b471, s0, "pagerankPruneDynamicFrontierOmp21", 0.0, batchFraction, batchIndex, 1e-17, 1e-17);
      auto b472 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R472, {repeat, tolerance, 1e-17, 1e-19});
      glog(b472, s0, "pagerankPruneDynamicFrontierOmp22", 0.0, batchFraction, batchIndex, 1e-17, 1e-19);
      auto b473 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R473, {repeat, tolerance, 1e-17, 1e-21});
      glog(b473, s0, "pagerankPruneDynamicFrontierOmp23", 0.0, batchFraction, batchIndex, 1e-17, 1e-21);
      auto b474 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R474, {repeat, tolerance, 1e-17, 1e-23});
      glog(b474, s0, "pagerankPruneDynamicFrontierOmp24", 0.0, batchFraction, batchIndex, 1e-17, 1e-23);
      auto b475 = pagerankPruneDynamicFrontierOmp<true, true, true, 2>(x, xt, y, yt, deletions, insertions, &R475, {repeat, tolerance, 1e-17, 1e-25});
      glog(b475, s0, "pagerankPruneDynamicFrontierOmp25", 0.0, batchFraction, batchIndex, 1e-17, 1e-25);
      copyValuesOmpW(R470, b470.ranks);
      copyValuesOmpW(R471, b471.ranks);
      copyValuesOmpW(R472, b472.ranks);
      copyValuesOmpW(R473, b473.ranks);
      copyValuesOmpW(R474, b474.ranks);
      copyValuesOmpW(R475, b475.ranks);
    }
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
      copyValuesOmpW(R550, c550.ranks);
      copyValuesOmpW(R551, c551.ranks);
      copyValuesOmpW(R552, c552.ranks);
      copyValuesOmpW(R553, c553.ranks);
      copyValuesOmpW(R554, c554.ranks);
      copyValuesOmpW(R555, c555.ranks);
    }
    {
      auto c560 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R560, {repeat, tolerance, 1e-7});
      glog(c560, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-7);
      auto c561 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R561, {repeat, tolerance, 1e-7, 1e-7});
      glog(c561, s0, "pagerankPruneDynamicFrontierOmp31", 0.0, batchFraction, batchIndex, 1e-7, 1e-7);
      auto c562 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R562, {repeat, tolerance, 1e-7, 1e-8});
      glog(c562, s0, "pagerankPruneDynamicFrontierOmp32", 0.0, batchFraction, batchIndex, 1e-7, 1e-8);
      auto c563 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R563, {repeat, tolerance, 1e-7, 1e-9});
      glog(c563, s0, "pagerankPruneDynamicFrontierOmp33", 0.0, batchFraction, batchIndex, 1e-7, 1e-9);
      auto c564 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R564, {repeat, tolerance, 1e-7, 1e-10});
      glog(c564, s0, "pagerankPruneDynamicFrontierOmp34", 0.0, batchFraction, batchIndex, 1e-7, 1e-10);
      auto c565 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R565, {repeat, tolerance, 1e-7, 1e-11});
      glog(c565, s0, "pagerankPruneDynamicFrontierOmp35", 0.0, batchFraction, batchIndex, 1e-7, 1e-11);
      copyValuesOmpW(R560, c560.ranks);
      copyValuesOmpW(R561, c561.ranks);
      copyValuesOmpW(R562, c562.ranks);
      copyValuesOmpW(R563, c563.ranks);
      copyValuesOmpW(R564, c564.ranks);
      copyValuesOmpW(R565, c565.ranks);
    }
    {
      auto c570 = pagerankDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R570, {repeat, tolerance, 1e-8});
      glog(c570, s0, "pagerankDynamicFrontierOmp3", 0.0, batchFraction, batchIndex, 1e-8);
      auto c571 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R571, {repeat, tolerance, 1e-8, 1e-8});
      glog(c571, s0, "pagerankPruneDynamicFrontierOmp31", 0.0, batchFraction, batchIndex, 1e-8, 1e-8);
      auto c572 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R572, {repeat, tolerance, 1e-8, 1e-9});
      glog(c572, s0, "pagerankPruneDynamicFrontierOmp32", 0.0, batchFraction, batchIndex, 1e-8, 1e-9);
      auto c573 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R573, {repeat, tolerance, 1e-8, 1e-10});
      glog(c573, s0, "pagerankPruneDynamicFrontierOmp33", 0.0, batchFraction, batchIndex, 1e-8, 1e-10);
      auto c574 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R574, {repeat, tolerance, 1e-8, 1e-11});
      glog(c574, s0, "pagerankPruneDynamicFrontierOmp34", 0.0, batchFraction, batchIndex, 1e-8, 1e-11);
      auto c575 = pagerankPruneDynamicFrontierOmp<true, true, true, 3>(x, xt, y, yt, deletions, insertions, &R575, {repeat, tolerance, 1e-8, 1e-12});
      glog(c575, s0, "pagerankPruneDynamicFrontierOmp35", 0.0, batchFraction, batchIndex, 1e-8, 1e-12);
      copyValuesOmpW(R570, c570.ranks);
      copyValuesOmpW(R571, c571.ranks);
      copyValuesOmpW(R572, c572.ranks);
      copyValuesOmpW(R573, c573.ranks);
      copyValuesOmpW(R574, c574.ranks);
      copyValuesOmpW(R575, c575.ranks);
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
