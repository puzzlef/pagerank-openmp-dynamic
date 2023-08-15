#pragma once
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include "_main.hxx"

using std::tuple;
using std::vector;
using std::get;
using std::abs;
using std::max;




#pragma region METHODS
#pragma region CALCULATE RANKS
/**
 * Calculate rank for a given vertex.
 * @param a current rank of each vertex (updated)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @returns change between previous and current rank value
 */
template <class H, class K, class V>
inline V pagerankPruneCalculateRank(vector<V>& a, const H& xt, const vector<V>& r, K v, V C0, V P) {
  V av = V();
  V rv = r[v];
  xt.forEachEdgeKey(v, [&](auto u) {
    K d = xt.vertexValue(u);
    av += r[u]/d;
  });
  av   = C0 + P * av;
  a[v] = av;
  return abs(av - rv);
}


/**
 * Calculate ranks for vertices in a graph.
 * @param a current rank of each vertex (updated)
 * @param vafe affected vertices for next iteration (output)
 * @param x original graph
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param vaff affected vertices for current iteration
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 */
template <bool ASYNC=false, class G, class H, class V, class B>
inline void pagerankPruneCalculateRanks(vector<V>& a, vector<B>& vafe, const G& x, const H& xt, const vector<V>& r, const vector<B>& vaff, V C0, V P, V D) {
  xt.forEachVertexKey([&](auto u) {
    if (!vaff[u]) return;
    V ev = pagerankPruneCalculateRank(a, xt, r, u, C0, P);
    if (ev>D) x.forEachEdgeKey(u, [&](auto v) { if (v!=u) vafe[v] = B(1); });
    if (ASYNC) vafe[u] = B(0);
  });
}


#ifdef OPENMP
/**
 * Calculate ranks for vertices in a graph (using OpenMP).
 * @param a current rank of each vertex (updated)
 * @param vafe affected vertices for next iteration (output)
 * @param x original graph
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param vaff affected vertices for current iteration
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param D frontier tolerance
 */
template <bool ASYNC=false, class G, class H, class V, class B>
inline void pagerankPruneCalculateRanksOmp(vector<V>& a, vector<B>& vafe, const G& x, const H& xt, const vector<V>& r, const vector<B>& vaff, V C0, V P, V D) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K u=0; u<S; ++u) {
    if (!xt.hasVertex(u) || !vaff[u]) continue;
    V ev = pagerankPruneCalculateRank(a, xt, r, u, C0, P);
    if (ev>D) x.forEachEdgeKey(u, [&](auto v) { if (v!=u) vafe[v] = B(1); });
    if (ASYNC) vafe[u] = B(0);
  }
}
#endif
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup environment and find the rank of each vertex in a graph.
 * @param x original graph
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V, class FL>
inline PagerankResult<V> pagerankPruneInvoke(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  using  B = FLAG;
  size_t S = xt.span();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), r(S);
  vector<B> vafe(S), vaff(S);
  float t = measureDuration([&]() {
    if (q) pagerankInitializeRanksFrom<ASYNC>(a, r, xt, *q);
    else   pagerankInitializeRanks    <ASYNC>(a, r, xt);
    l = fl(ASYNC? r : a, r, ASYNC? vaff : vafe, vaff, x, xt, P, E, L);
  }, o.repeat);
  return {r, l, t};
}


#ifdef OPENMP
/**
 * Setup environment and find the rank of each vertex in a graph (using OpenMP).
 * @param x original graph
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V, class FL>
inline PagerankResult<V> pagerankPruneInvokeOmp(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  using  B = FLAG;
  size_t S = xt.span();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), r(S);
  vector<B> vafe(S), vaff(S);
  float t = measureDuration([&]() {
    if (q) pagerankInitializeRanksFromOmp<ASYNC>(a, r, xt, *q);
    else   pagerankInitializeRanksOmp    <ASYNC>(a, r, xt);
    l = fl(ASYNC? r : a, r, ASYNC? vaff : vafe, vaff, x, xt, P, E, L);
  }, o.repeat);
  return {r, l, t};
}
#endif
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform PageRank iterations upon a graph.
 * @param a current rank of each vertex (updated)
 * @param r previous rank of each vertex (updated)
 * @param vafe affected vertices for next iteration (updated)
 * @param vaff affected vertices for current iteration (updated)
 * @param x original graph
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @returns iterations performed
 */
template <bool ASYNC=false, class G, class H, class V, class B>
inline int pagerankPruneLoop(vector<V>& a, vector<V>& r, vector<B>& vafe, vector<B>& vaff, const G& x, const H& xt, V P, V E, int L) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  V  C0 = (1-P)/N;
  V   D = 0.001 * E;  // Frontier tolerance = Tolerance/1000
  while (l<L) {
    if (!ASYNC) fillValueU(vafe, B(0));   // Reset affected vertices for next iteration
    pagerankPruneCalculateRanks<ASYNC>(a, vafe, x, xt, r, vaff, C0, P, D); ++l;  // Update ranks of vertices
    V el = pagerankDeltaRanks(xt, a, r);  // Compare previous and current ranks
    if (!ASYNC) swap(vafe, vaff);         // Affected vertices in (vaff)
    if (!ASYNC) swap(a, r);               // Final ranks in (r)
    if (el<E) break;                      // Check tolerance
  }
  return l;
}


#ifdef OPENMP
/**
 * Perform PageRank iterations upon a graph (using OpenMP).
 * @param a current rank of each vertex (updated)
 * @param r previous rank of each vertex (updated)
 * @param vafe affected vertices for next iteration (updated)
 * @param vaff affected vertices for current iteration (updated)
 * @param x original graph
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @returns iterations performed
 */
template <bool ASYNC=false, class G, class H, class V, class B>
inline int pagerankPruneLoopOmp(vector<V>& a, vector<V>& r, vector<B>& vafe, vector<B>& vaff, const G& x, const H& xt, V P, V E, int L) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  V  C0 = (1-P)/N;
  V   D = 0.001 * E;  // Frontier tolerance = Tolerance/1000
  while (l<L) {
    if (!ASYNC) fillValueOmpU(vafe, B(0));   // Reset affected vertices for next iteration
    pagerankPruneCalculateRanksOmp<ASYNC>(a, vafe, x, xt, r, vaff, C0, P, D); ++l;  // Update ranks of vertices
    V el = pagerankDeltaRanksOmp(xt, a, r);  // Compare previous and current ranks
    if (!ASYNC) swap(vafe, vaff);            // Affected vertices in (vaff)
    if (!ASYNC) swap(a, r);                  // Final ranks in (r)
    if (el<E) break;                         // Check tolerance
  }
  return l;
}
#endif
#pragma endregion




#pragma region STATIC/NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param x original graph
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankPruneNaiveDynamic(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  using B = FLAG;
  if  (xt.empty()) return {};
  return pagerankPruneInvoke<ASYNC, FLAG>(x, xt, q, o, [&](vector<V>& a, vector<V>& r, vector<B>& vafe, vector<B>& vaff, const G& x, const H& xt, V P, V E, int L) {
    fillValueOmpU(vaff, B(1));
    return pagerankPruneLoop<ASYNC>(a, r, vafe, vaff, x, xt, P, E, L);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with Naive-dynamic approach.
 * @param x original graph
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class FLAG=char, class G, class H, class V>
inline PagerankResult<V> pagerankPruneNaiveDynamicOmp(const G& x, const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  using B = FLAG;
  if  (xt.empty()) return {};
  return pagerankPruneInvokeOmp<ASYNC, FLAG>(x, xt, q, o, [&](vector<V>& a, vector<V>& r, vector<B>& vafe, vector<B>& vaff, const G& x, const H& xt, V P, V E, int L) {
    fillValueOmpU(vaff, B(1));
    return pagerankPruneLoopOmp<ASYNC>(a, r, vafe, vaff, x, xt, P, E, L);
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
  using  B = FLAG;
  if (xt.empty()) return {};
  return pagerankPruneInvoke<ASYNC, FLAG>(y, yt, q, o, [&](vector<V>& a, vector<V>& r, vector<B>& vafe, vector<B>& vaff, const G& x, const H& xt, V P, V E, int L) {
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankPruneLoop<ASYNC>(a, r, vafe, vaff, x, xt, P, E, L);
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
  using  B = FLAG;
  if (xt.empty()) return {};
  return pagerankPruneInvokeOmp<ASYNC, FLAG>(y, yt, q, o, [&](vector<V>& a, vector<V>& r, vector<B>& vafe, vector<B>& vaff, const G& x, const H& xt, V P, V E, int L) {
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankPruneLoopOmp<ASYNC>(a, r, vafe, vaff, x, xt, P, E, L);
  });
}
#endif
#pragma endregion
#pragma endregion
