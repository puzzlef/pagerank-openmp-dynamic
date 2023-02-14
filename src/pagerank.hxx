#pragma once
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include "_main.hxx"
#include "csr.hxx"
#include "vertices.hxx"
#include "transpose.hxx"
#include "dfs.hxx"
#include "components.hxx"

#ifdef OPENMP
#include <omp.h>
#endif

using std::tuple;
using std::vector;
using std::move;
using std::abs;
using std::max;




// PAGERANK OPTIONS
// ----------------

enum NormFunction {
  L0_NORM = 0,
  L1_NORM = 1,
  L2_NORM = 2,
  LI_NORM = 3
};


template <class V>
struct PagerankOptions {
  int repeat;
  int toleranceNorm;
  V   tolerance;
  V   damping;
  int maxIterations;

  PagerankOptions(int repeat=1, int toleranceNorm=LI_NORM, V tolerance=1e-10, V damping=0.85, int maxIterations=500) :
  repeat(repeat), toleranceNorm(toleranceNorm), tolerance(tolerance), damping(damping), maxIterations(maxIterations) {}
};




// PAGERANK RESULT
// ---------------

template <class V>
struct PagerankResult {
  vector<V> ranks;
  int   iterations;
  float time;

  PagerankResult() :
  ranks(), iterations(0), time(0) {}

  PagerankResult(vector<V>&& ranks, int iterations=0, float time=0) :
  ranks(ranks), iterations(iterations), time(time) {}

  PagerankResult(vector<V>& ranks, int iterations=0, float time=0) :
  ranks(move(ranks)), iterations(iterations), time(time) {}
};




// PAGERANK INITIALIZE
// -------------------
// Initialize ranks of each vertex.

/**
 * Initialize rank of each vertex to 1/N.
 * @param a rank of each vertex (output)
 * @param xt transpose of original graph
 */
template <class H, class V>
inline void pagerankInitialize(vector<V>& a, const H& xt) {
  size_t N = xt.order();
  xt.forEachVertexKey([&](auto v) {
    a[v] = V(1)/N;
  });
}

/**
 * Initialize rank of each vertex from given initial ranks.
 * @param a rank of each vertex (output)
 * @param xt transpose of original graph
 * @param q initial rank of each vertex
 */
template <class H, class V>
inline void pagerankInitializeFrom(vector<V>& a, const H& xt, const vector<V>& q) {
  xt.forEachVertexKey([&](auto v) {
    a[v] = q[v];
  });
}


#ifdef OPENMP
template <class H, class V>
inline void pagerankInitializeOmp(vector<V>& a, const H& xt) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v)) continue;
    a[v] = V(1)/N;
  }
}

template <class H, class V>
inline void pagerankInitializeFromOmp(vector<V>& a, const H& xt, const vector<V>& q) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v)) continue;
    a[v] = q[v];
  }
}
#endif




// PAGERANK FACTOR
// ---------------
// For contribution factors of vertices (unchanging).

/**
 * Calculate rank scaling factor for each vertex.
 * @param a rank scaling factor for each vertex (output)
 * @param xt transpose of original graph
 * @param P damping factor [0.85]
 */
template <class H, class V>
inline void pagerankFactor(vector<V>& a, const H& xt, V P) {
  xt.forEachVertex([&](auto v, auto d) {
    a[v] = d>0? P/d : 0;
  });
}


#ifdef OPENMP
template <class H, class V>
inline void pagerankFactorOmp(vector<V>& a, const H& xt, V P) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  #pragma omp parallel for schedule(auto)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v)) continue;
    K  d = xt.vertexData(v);
    a[v] = d>0? P/d : 0;
  }
}
#endif




// PAGERANK TELEPORT
// -----------------
// For teleport contribution from vertices (inc. dead ends).

/**
 * Find total teleport contribution from each vertex (inc. deade ends).
 * @param xt transpose of original graph
 * @param r rank of each vertex
 * @param P damping factor [0.85]
 * @returns common teleport rank contribution to each vertex
 */
template <class H, class V>
inline V pagerankTeleport(const H& xt, const vector<V>& r, V P) {
  size_t N = xt.order();
  V a = (1-P)/N;
  xt.forEachVertex([&](auto v, auto d) {
    if (d==0) a += P * r[v]/N;
  });
  return a;
}


#ifdef OPENMP
template <class H, class V>
inline V pagerankTeleportOmp(const H& xt, const vector<V>& r, V P) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V a = (1-P)/N;
  #pragma omp parallel for schedule(auto) reduction(+:a)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v)) continue;
    K   d = xt.vertexData(v);
    if (d==0) a += P * r[v]/N;
  }
  return a;
}
#endif




// PAGERANK CALCULATE
// ------------------
// For rank calculation from in-edges.

/**
 * Calculate rank for a given vertex.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param c rank contribution from each vertex
 * @param v given vertex
 * @param C0 common teleport rank contribution to each vertex
 * @returns change in rank
 */
template <class H, class K, class V>
inline V pagerankCalculateRankDelta(vector<V>& a, const H& xt, const vector<V>& r, const vector<V>& c, K v, V C0) {
  V av = C0;
  V rv = r[v];
  xt.forEachEdgeKey([&](auto u) {
    av += c[u];
  });
  a[v] = av;
  return av - rv;
}


/**
 * Calculate ranks for vertices in a graph.
 * @param a current rank of each vertex (output)
 * @param xt transpose of original graph
 * @param r previous rank of each vertex
 * @param c rank contribution from each vertex
 * @param C0 common teleport rank contribution to each vertex
 * @param fa is vertex affected? (vertex)
 * @param fp per vertex processing (vertex)
 */
template <class H, class K, class V, class FA, class FP>
inline void pagerankCalculateRanks(vector<V>& a, const H& xt, const vector<V>& r, const vector<V>& c, V C0, V E, FA fa, FP fp) {
  xt.forEachVertexKey([&](auto v) {
    if (!fa(v)) continue;
    if (abs(pagerankCalculateRankDelta(a, xt, r, c, v, C0)) > E) fp(v);
  });
}


#ifdef OPENMP
template <class H, class K, class V, class FA, class FP>
inline void pagerankCalculateRanksOmp(vector<V>& a, const H& xt, const vector<V>& r, const vector<V>& c, V C0, V E, FA fa, FP fp) {
  size_t S = xt.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K v=0; v<S; ++v) {
    if (!xt.hasVertex(v) || !fa(v)) continue;
    if (abs(pagerankCalculateRankDelta(a, xt, r, c, v, C0)) > E) fp(v);
  }
}
#endif




// PAGERANK ERROR
// --------------
// For convergence check.

/**
 * Get the error between two rank vectors.
 * @param x first rank vector
 * @param y second rank vector
 * @param EF error function (L1/L2/LI)
 * @returns error between the two rank vectors
 */
template <class V>
inline V pagerankError(const vector<V>& x, const vector<V>& y, int EF) {
  switch (EF) {
    case 1:  return l1Norm(x, y);
    case 2:  return l2Norm(x, y);
    default: return liNorm(x, y);
  }
}


#ifdef OPENMP
template <class V>
inline V pagerankErrorOmp(const vector<V>& x, const vector<V>& y, int EF) {
  switch (EF) {
    case 1:  return l1NormOmp(x, y);
    case 2:  return l2NormOmp(x, y);
    default: return liNormOmp(x, y);
  }
}
#endif




// PAGERANK AFFECTED (TRAVERSAL)
// -----------------------------

/**
 * Find affected vertices due to a batch update.
 * @param x original graph
 * @param y updated graph
 * @param ft is vertex affected? (u)
 * @returns affected flags
 */
template <class G, class FT>
inline auto pagerankAffectedTraversal(const G& x, const G& y, FT ft) {
  auto fn = [](auto u) {};
  vector<bool> vis(max(x.span(), y.span()));
  y.forEachVertexKey([&](auto u) {
    if (!ft(u)) return;
    dfsVisitedForEachW(vis, x, u, fn);
    dfsVisitedForEachW(vis, y, u, fn);
  });
  return vis;
}


/**
 * Find affected vertices due to a batch update.
 * @param y original graph
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @returns affected flags
 */
template <class G, class K>
inline auto pagerankAffectedTraversal(const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
  auto fn = [](K u) {};
  vector<bool> vis(max(x.span(), y.span()));
  for (const auto& [u, v] : deletions)
    dfsVisitedForEachW(vis, x, u, fn);
  for (const auto& [u, v] : insertions)
    dfsVisitedForEachW(vis, y, u, fn);
  return vis;
}




// PAGERANK AFFECTED (FRONTIER)
// ----------------------------

/**
 * Find affected vertices due to a batch update.
 * @param x original graph
 * @param y updated graph
 * @param ft is vertex affected? (u)
 * @returns affected flags
 */
template <class G, class FT>
inline auto pagerankAffectedFrontier(const G& x, const G& y, FT ft) {
  auto fn = [](auto u) {};
  vector<bool> vis(max(x.span(), y.span()));
  y.forEachVertexKey([&](auto u) {
    if (!ft(u)) return;
    vis[u] = true;
    x.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
    y.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
  });
  return vis;
}


/**
 * Find affected vertices due to a batch update.
 * @param y original graph
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @returns affected flags
 */
template <class G, class K>
inline auto pagerankAffectedFrontier(const G& x, const G& y, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions) {
  auto fn = [](K u) {};
  vector<bool> vis(max(x.span(), y.span()));
  for (const auto& [u, v] : deletions) {
    vis[u] = true;
    x.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
  }
  for (const auto& [u, v] : insertions) {
    vis[u] = true;
    y.forEachEdgeKey(u, [&](auto v) { vis[v] = true; });
  }
  return vis;
}




// PAGERANK-SEQ
// ------------
// For single-threaded (sequential) PageRank implementation.

/**
 * Find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @param fa is vertex affected? (vertex)
 * @param fp per vertex processing (vertex)
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V, class FL, class FA, class FP>
PagerankResult<V> pagerankSeq(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl, FA fa, FP fp) {
  using  K = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  int EF = o.toleranceNorm;
  vector<int> e(S); vector<V> a(S), r(S), c(S), f(S);
  float t = measureDuration([&]() {
    fillValueU(e, 0);
    if (q) pagerankInitializeFrom(r, xt, *q);
    else   pagerankInitialize    (r, xt);
    if (!ASYNC) copyValuesW(a, r);
    pagerankFactor(f, xt, P); multiplyValuesW(c, r, f);  // calculate factors (f) and contributions (c)
    l = fl(e, ASYNC? r : a, r, c, f, xt, P, E, L, EF, fa, fp);  // calculate ranks of vertices
  }, o.repeat);
  return {r, l, t};
}




// PAGERANK-OMP
// ------------
// For multi-threaded OpenMP-based PageRank implementation.

#ifdef OPENMP
/**
 * Find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @param fl update loop
 * @param fa is vertex affected? (vertex)
 * @param fp per vertex processing (vertex)
 * @returns pagerank result
 */
template <bool ASYNC=false, class H, class V, class FL, class FA, class FP>
PagerankResult<V> pagerankOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o, FL fl, FA fa, FP fp) {
  using K  = typename H::key_type;
  size_t S = xt.span();
  size_t N = xt.order();
  V   P  = o.damping;
  V   E  = o.tolerance;
  int L  = o.maxIterations, l = 0;
  int EF = o.toleranceNorm;
  vector<int> e(S); vector<V> a(S), r(S), c(S), f(S);
  float t = measureDuration([&]() {
    fillValueU(e, 0);
    if (q) pagerankInitializeFromOmp(r, xt, *q);
    else   pagerankInitializeOmp    (r, xt);
    if (!ASYNC) copyValuesOmpW(a, r);
    pagerankFactorOmp(f, xt, P); multiplyValuesOmpW(c, r, f);  // calculate factors (f) and contributions (c)
    l = fl(e, ASYNC? r : a, r, c, f, xt, P, E, L, EF, fa, fp);  // calculate ranks of vertices
  }, o.repeat);
  return {r, l, t};
}
#endif
