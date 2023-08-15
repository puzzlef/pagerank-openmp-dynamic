#pragma once
#include <tuple>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "pagerank.hxx"

using std::tuple;
using std::vector;
using std::max;




#pragma region METHODS
#pragma region STATIC/NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankPruneNaiveDynamic(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  V D = 0.001 * o.tolerance;  // Frontier tolerance = Tolerance/1000
  vector<FLAG> vaff(xt.span());
  return pagerankInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) {
      vaff[u] = FLAG(0);
      if (eu>D) x.forEachEdgeKey(u, [&](K v) { if (v!=u) vaff[v] = FLAG(1); });
    };
    fillValueU(vaff, FLAG(1));
    return pagerankLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankPruneNaiveDynamicOmp(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  V D = 0.001 * o.tolerance;  // Frontier tolerance = Tolerance/1000
  vector<FLAG> vaff(xt.span());
  return pagerankInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) {
      vaff[u] = FLAG(0);
      if (eu>D) x.forEachEdgeKey(u, [&](K v) { if (v!=u) vaff[v] = FLAG(1); });
    };
    fillValueOmpU(vaff, FLAG(1));
    return pagerankLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion




#pragma region DYNAMIC FRONTIER
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Frontier approach.
 * @param x original graph
 * @param xt transpose of original graph
 * @param y updated graph
 * @param yt transpose of updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class K, class V, class W>
inline PagerankResult<V> pagerankPruneDynamicFrontier(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, W>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // Frontier tolerance = Tolerance/1000
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankInvoke<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) {
      vaff[u] = FLAG(0);
      if (eu>D) y.forEachEdgeKey(u, [&](K v) { if (v!=u) vaff[v] = FLAG(1); });
    };
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankLoop<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with Dynamic Frontier approach (using OpenMP).
 * @param x original graph
 * @param xt transpose of original graph
 * @param y updated graph
 * @param yt transpose of updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class K, class V, class W>
inline PagerankResult<V> pagerankPruneDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, W>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // Frontier tolerance = Tolerance/1000
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankInvokeOmp<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& r, const H& xt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fr = [&](K u, V eu) {
      vaff[u] = FLAG(0);
      if (eu>D) y.forEachEdgeKey(u, [&](K v) { if (v!=u) vaff[v] = FLAG(1); });
    };
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankLoopOmp<ASYNC>(a, r, xt, P, E, L, fa, fr);
  });
}
#endif
#pragma endregion
#pragma endregion
