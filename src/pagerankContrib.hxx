#pragma once
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>
#include "_main.hxx"
#include "pagerank.hxx"

using std::tuple;
using std::vector;
using std::abs;
using std::max;




#pragma region METHODS
#pragma region CALCULATE CONTRIBUTIONS
/**
 * Calculate contribution for a given vertex.
 * @param a current contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param c previous contribution of each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @returns change between previous and current contribution value
 */
template <class H, class K, class V>
inline V pagerankCalculateContrib(vector<V>& a, const H& xt, const vector<V>& c, K v, V C0, V P) {
  V av = V();
  V cv = c[v];
  K  d = xt.vertexValue(v);
  xt.forEachEdgeKey(v, [&](auto u) { av += c[u]; });
  av   = (C0 + P * av)/d;
  a[v] = av;
  return abs(av - cv);
}


/**
 * Calculate contributions for vertices in a graph.
 * @param a current contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param c previous contribution of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fc called with vertex contribution change (vertex, delta)
 */
template <class H, class V, class FA, class FC>
inline void pagerankCalculateContribs(vector<V>& a, const H& xt, const vector<V>& c, V C0, V P, FA fa, FC fc) {
  xt.forEachVertexKey([&](auto v) {
    if (!fa(v)) return;
    V ev = pagerankCalculateContrib(a, xt, c, v, C0, P);
    fc(v, ev);
  });
}


#ifdef OPENMP
/**
 * Calculate contributions for vertices in a graph (using OpenMP).
 * @param a current contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param c previous contribution of each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param P damping factor [0.85]
 * @param fa is vertex affected? (vertex)
 * @param fc called with vertex contribution change (vertex, delta)
 */
template <class H, class V, class FA, class FC>
inline void pagerankCalculateContribsOmp(vector<V>& a, const H& xt, const vector<V>& c, V C0, V P, FA fa, FC fc) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    V ev = pagerankCalculateContrib(a, xt, c, v, C0, P);
    fc(v, ev);
  }
}
#endif
#pragma endregion




#pragma region INITIALIZE CONTRIBUTIONS
/**
 * Intitialize contributions before PageRank iterations.
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeContribs(vector<V>& a, vector<V>& c, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  for (K v=0; v<S; ++v) {
    K  d = xt.vertexValue(v);
    c[v] = xt.hasVertex(v)? (V(1)/d)/N : V();
    if (!ASYNC) a[v] = c[v];
  }
}


#ifdef OPENMP
/**
 * Intitialize contributions before PageRank iterations (using OpenMP).
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param xt transpose of original graph
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeContribsOmp(vector<V>& a, vector<V>& c, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    K  d = xt.vertexValue(v);
    c[v] = xt.hasVertex(v)? (V(1)/d)/N : V();
    if (!ASYNC) a[v] = c[v];
  }
}
#endif


/**
 * Intitialize contributions before PageRank iterations from given ranks.
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param q initial ranks
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeContribsFrom(vector<V>& a, vector<V>& c, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  for (K v=0; v<S; ++v) {
    K  d = xt.vertexValue(v);
    c[v] = q[v] / d;
    if (!ASYNC) a[v] = c[v];
  }
}


#ifdef OPENMP
/**
 * Intitialize contributions before PageRank iterations from given ranks (using OpenMP).
 * @param a current contribution of each vertex (output)
 * @param c previous contribution of each vertex (output)
 * @param xt transpose of original graph
 * @param q initial ranks
 */
template <bool ASYNC=false, class H, class V>
inline void pagerankInitializeContribsFromOmp(vector<V>& a, vector<V>& c, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    K  d = xt.vertexValue(v);
    c[v] = q[v] / d;
    if (!ASYNC) a[v] = c[v];
  }
}
#endif
#pragma endregion




#pragma region CONVERT CONTRIBUTIONS (TO RANKS)
/**
 * Convert contributions to ranks.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param c current contribution of each vertex
 */
template <class H, class V>
inline void pagerankContribRanks(vector<V>& a, const H& xt, const vector<V>& c) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  for (K v=0; v<S; ++v) {
    K  d = xt.vertexValue(v);
    a[v] = xt.hasVertex(v)? d * c[v] : V();
  }
}


#ifdef OPENMP
/**
 * Convert contributions to ranks (using OpenMP).
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param c current contribution of each vertex
 */
template <class H, class V>
inline void pagerankContribRanksOmp(vector<V>& a, const H& xt, const vector<V>& c) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    K  d = xt.vertexValue(v);
    a[v] = xt.hasVertex(v)? d * c[v] : V();
  }
}
#endif
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup environment and find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V, class FL>
inline PagerankResult<V> pagerankContribInvoke(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), c(S);
  float t = measureDuration([&]() {
    if (q) pagerankInitializeContribsFrom<ASYNC>(a, c, xt, *q);
    else   pagerankInitializeContribs    <ASYNC>(a, c, xt);
    l = fl(ASYNC? c : a, c, xt, P, E, L);
  }, o.repeat);
  pagerankContribRanks(a, xt, c);
  return {a, l, t};
}




#ifdef OPENMP
/**
 * Setup environment and find the rank of each vertex in a graph (using OpenMP).
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V, class FL>
inline PagerankResult<V> pagerankContribInvokeOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  vector<V> a(S), c(S);
  float t = measureDuration([&]() {
    if (q) pagerankInitializeContribsFromOmp<ASYNC>(a, c, xt, *q);
    else   pagerankInitializeContribsOmp    <ASYNC>(a, c, xt);
    l = fl(ASYNC? c : a, c, xt, P, E, L);
  }, o.repeat);
  pagerankContribRanksOmp(a, xt, c);
  return {a, l, t};
}
#endif
#pragma endregion




#pragma region DELTA CONTRIBUTIONS
/**
 * Calculate rank delta between two contribution vectors.
 * @param xt transpose of original graph
 * @param a current contribution of each vertex
 * @param c previous contribution of each vertex
 * @returns ||a//D - c//D||_∞
 */
template <class H, class V>
inline V pagerankDeltaContribs(const H& xt, const vector<V>& a, const vector<V>& c) {
  using K = typename H::key_type;
  V e = V();
  xt.forEachVertexKey([&](auto v) {
    K d  = xt.vertexValue(v);
    V av = d * a[v];
    V rv = d * c[v];
    e = max(e, abs(av - rv));
  });
  return e;
}


#ifdef OPENMP
/**
 * Calculate rank delta between two contribution vectors (using OpenMP).
 * @param xt transpose of original graph
 * @param a current contribution of each vertex
 * @param c previous contribution of each vertex
 * @returns ||a//D - c//D||_∞
 */
template <class H, class V>
inline V pagerankDeltaContribsOmp(const H& xt, const vector<V>& a, const vector<V>& c) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  V e = V();
  #pragma omp parallel for schedule(auto) reduction(max:e)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v)) continue;
    K d  = xt.vertexValue(v);
    V av = d * a[v];
    V rv = d * c[v];
    e = max(e, abs(av - rv));
  }
  return e;
}
#endif
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform PageRank iterations upon a graph.
 * @param a current contribution of each vertex (updated)
 * @param c previous contribution of each vertex (updated)
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param fa is vertex affected? (vertex)
 * @param fc called if vertex contribution changes (vertex, delta)
 * @returns iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FC>
inline int pagerankContribLoop(vector<V>& a, vector<V>& c, const H& xt, V P, V E, int L, FA fa, FC fc) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  V  C0 = (1-P)/N;
  while (l<L) {
    pagerankCalculateContribs(a, xt, c, C0, P, fa, fc); ++l;  // Update contributions of vertices
    V el = pagerankDeltaContribs(xt, a, c);  // Compare previous and current contributions
    if (!ASYNC) swap(a, c);                  // Final contributions in (c)
    if (el<E) break;                         // Check tolerance
  }
  return l;
}


#ifdef OPENMP
/**
 * Perform PageRank iterations upon a graph (using OpenMP).
 * @param a current contribution of each vertex (updated)
 * @param c previous contribution of each vertex (updated)
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param fa is vertex affected? (vertex)
 * @param fc called if vertex contribution changes (vertex, delta)
 * @returns iterations performed
 */
template <bool ASYNC=false, class H, class V, class FA, class FC>
inline int pagerankContribLoopOmp(vector<V>& a, vector<V>& c, const H& xt, V P, V E, int L, FA fa, FC fc) {
  using  K = typename H::key_type;
  size_t N = xt.order();
  int l = 0;
  V  C0 = (1-P)/N;
  while (l<L) {
    pagerankCalculateContribsOmp(a, xt, c, C0, P, fa, fc); ++l;  // Update contributions of vertices
    V el = pagerankDeltaContribsOmp(xt, a, c);  // Compare previous and current contributions
    if (!ASYNC) swap(a, c);                     // Final contributions in (c)
    if (el<E) break;                            // Check tolerance
  }
  return l;
}
#endif
#pragma endregion




#pragma region STATIC/NAIVE-DYNAMIC
/**
 * Find the rank of each vertex in a static graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankContrib(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  return pagerankContribInvoke<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& c, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fc = [](K u, V eu) {};
    return pagerankContribLoop<ASYNC>(a, c, xt, P, E, L, fa, fc);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a static graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V>
inline PagerankResult<V> pagerankContribOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  if  (xt.empty()) return {};
  return pagerankContribInvokeOmp<ASYNC>(xt, q, o, [&](vector<V>& a, vector<V>& c, const H& xt, V P, V E, int L) {
    auto fa = [](K u) { return true; };
    auto fc = [](K u, V eu) {};
    return pagerankContribLoopOmp<ASYNC>(a, c, xt, P, E, L, fa, fc);
  });
}
#endif
#pragma endregion




#pragma region DYNAMIC FRONTIER
/**
 * Find the rank of each vertex in a dynamic graph with the Dynamic Frontier approach.
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
inline PagerankResult<V> pagerankContribDynamicFrontier(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, W>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // Frontier tolerance = Tolerance/1000
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankContribInvoke<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& c, const H& yt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fc = [&](K u, V eu) { if (eu * yt.vertexValue(u) > D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierW(vaff, x, y, deletions, insertions);
    return pagerankContribLoop<ASYNC>(a, c, yt, P, E, L, fa, fc);
  });
}


#ifdef OPENMP
/**
 * Find the rank of each vertex in a dynamic graph with the Dynamic Frontier approach (using OpenMP).
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
inline PagerankResult<V> pagerankContribDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K, W>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  V D = 0.001 * o.tolerance;  // Frontier tolerance = Tolerance/1000
  if (xt.empty()) return {};
  vector<FLAG> vaff(max(x.span(), y.span()));
  return pagerankContribInvokeOmp<ASYNC>(yt, q, o, [&](vector<V>& a, vector<V>& c, const H& yt, V P, V E, int L) {
    auto fa = [&](K u) { return vaff[u]==FLAG(1); };
    auto fc = [&](K u, V eu) { if (eu * yt.vertexValue(u) > D) y.forEachEdgeKey(u, [&](K v) { vaff[v] = FLAG(1); }); };
    pagerankAffectedFrontierOmpW(vaff, x, y, deletions, insertions);
    return pagerankContribLoopOmp<ASYNC>(a, c, yt, P, E, L, fa, fc);
  });
}
#endif
#pragma endregion
#pragma endregion
