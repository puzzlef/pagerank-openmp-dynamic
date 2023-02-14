#pragma once
#include <utility>
#include <algorithm>
#include <vector>
#include "_main.hxx"
#include "vertices.hxx"
#include "pagerank.hxx"

using std::tuple;
using std::vector;
using std::swap;




// PAGERANK LOOP
// -------------

/**
 * Perform PageRank iterations upon a graph.
 * @param e change in rank for each vertex below tolerance? (unused)
 * @param a current rank of each vertex (updated)
 * @param r previous rank of each vertex (updated)
 * @param c rank contribution from each vertex (updated)
 * @param f rank scaling factor for each vertex
 * @param xv edge offsets for each vertex in the graph
 * @param xe target vertices for each edge in the graph
 * @param vdeg out-degree of each vertex
 * @param N total number of vertices
 * @param P damping factor [0.85]
 * @param E tolerance [10^-10]
 * @param L max. iterations [500]
 * @param EF error function (L1/L2/LI)
 * @param i vertex start
 * @param n vertex count
 * @param threads information on each thread (updated)
 * @param fv per vertex processing (thread, vertex)
 * @param fa is vertex affected? (vertex)
 * @returns iterations performed
 */
template <bool ASYNC=false, bool DEAD=false, class K, class V, class FA, class FP>
inline int pagerankBasicSeqLoop(vector<int>& e, vector<V>& a, vector<V>& r, vector<V>& c, const vector<V>& f, const vector<size_t>& xv, const vector<K>& xe, const vector<K>& vdeg, K N, V P, V E, int L, int EF, K i, K n, FA fa, FP fp) {
  int l = 0;
  while (l<L) {
    V C0 = DEAD? pagerankTeleport(r, vdeg, P, N) : (1-P)/N;
    pagerankCalculateRanks(a, r, c, xv, xe, C0, E, i, n, fa, fp); ++l;  // update ranks of vertices
    multiplyValuesW(c, a, f, i, n);        // update partial contributions (c)
    V el = pagerankError(a, r, EF, i, n);  // compare previous and current ranks
    if (!ASYNC) swap(a, r);                // final ranks in (r)
    if (el<E) break;                       // check tolerance
  }
  return l;
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class K, class V, class FA, class FP>
inline int pagerankBasicOmpLoop(vector<int>& e, vector<V>& a, vector<V>& r, vector<V>& c, const vector<V>& f, const vector<size_t>& xv, const vector<K>& xe, const vector<K>& vdeg, K N, V P, V E, int L, int EF, K i, K n, FA fa, FP fp) {
  int l = 0;
  while (l<L) {
    V C0 = DEAD? pagerankTeleportOmp(r, vdeg, P, N) : (1-P)/N;
    pagerankCalculateRanksOmp(a, r, c, xv, xe, C0, E, i, n, fa, fp); ++l;  // update ranks of vertices
    multiplyValuesOmpW(c, a, f, i, n);        // update partial contributions (c)
    V el = pagerankErrorOmp(a, r, EF, i, n);  // compare previous and current ranks
    if (!ASYNC) swap(a, r);                   // final ranks in (r)
    if (el<E) break;                          // check tolerance
  }
  return l;
}
#endif




// STATIC/NAIVE-DYNAMIC PAGERANK
// -----------------------------

/**
 * Find the rank of each vertex in a graph.
 * @param xt transpose of original graph
 * @param q initial ranks
 * @param o pagerank options
 * @returns pagerank result
 */
template <bool ASYNC=false, bool DEAD=false, class H, class V>
inline PagerankResult<V> pagerankBasicSeq(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  K    N  = xt.order();  if (N==0) return {};
  auto ks = vertexKeys(xt);
  auto fa = [](K u) { return true; };
  auto fp = [](K u) {};
  return pagerankSeq<ASYNC>(xt, q, o, ks, 0, N, pagerankBasicSeqLoop<ASYNC, DEAD, K, V, decltype(fa), decltype(fp)>, fa, fp);
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class H, class V>
inline PagerankResult<V> pagerankBasicOmp(const H& xt, const vector<V> *q, const PagerankOptions<V>& o) {
  using K = typename H::key_type;
  K    N  = xt.order();  if (N==0) return {};
  auto ks = vertexKeys(xt);
  auto fa = [](K u) { return true; };
  auto fp = [](K u) {};
  return pagerankOmp<ASYNC>(xt, q, o, ks, 0, N, pagerankBasicOmpLoop<ASYNC, DEAD, K, V, decltype(fa), decltype(fp)>, fa, fp);
}
#endif




// TRAVERSAL-BASED DYNAMIC PAGERANK
// --------------------------------

/**
 * Find the rank of each vertex in a dynamic graph.
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
template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V>
inline PagerankResult<V> pagerankBasicDynamicTraversalSeq(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  K    N    = yt.order();  if (N==0) return {};
  auto ks   = vertexKeys(yt);
  auto vaff = compressContainer(y, pagerankAffectedTraversal(x, y, deletions, insertions), ks);
  auto fa   = [&](K u) { return vaff[u]==true; };
  auto fp   = [ ](K u) {};
  return pagerankSeq<ASYNC>(yt, q, o, ks, 0, N, pagerankBasicSeqLoop<ASYNC, DEAD, K, V, decltype(fa), decltype(fp)>, fa, fp);
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V, class FV>
inline PagerankResult<V> pagerankBasicDynamicTraversalOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  K    N    = yt.order();  if (N==0) return {};
  auto ks   = vertexKeys(yt);
  auto vaff = compressContainer(y, pagerankAffectedTraversal(x, y, deletions, insertions), ks);
  auto fa   = [&](K u) { return vaff[u]==true; };
  auto fp   = [ ](K u) {};
  return pagerankOmp<ASYNC>(yt, q, o, ks, 0, N, pagerankBasicOmpLoop<ASYNC, DEAD, K, V, decltype(fa), decltype(fp)>, fa, fp);
}
#endif




// FRONTIER-BASED DYNAMIC PAGERANK
// -------------------------------

/**
 * Find the rank of each vertex in a dynamic graph.
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
template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V>
inline PagerankResult<V> pagerankBasicDynamicFrontierSeq(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  K    N    = yt.order();  if (N==0) return {};
  auto ks   = vertexKeys(yt);
  auto vaff = compressContainer(y, pagerankAffectedFrontier(x, y, deletions, insertions), ks);
  auto fa   = [&](K u) { return vaff[u]==true; };
  auto fp   = [&](K u) { y.forEachEdgeKey(u, [&](auto v) { vaff[v] = true; }); };
  return pagerankSeq<ASYNC>(yt, q, o, ks, 0, N, pagerankBasicSeqLoop<ASYNC, DEAD, K, V, decltype(fa), decltype(fp)>, fa, fp);
}


#ifdef OPENMP
template <bool ASYNC=false, bool DEAD=false, class G, class H, class K, class V, class FV>
inline PagerankResult<V> pagerankBasicDynamicFrontierOmp(const G& x, const H& xt, const G& y, const H& yt, const vector<tuple<K, K>>& deletions, const vector<tuple<K, K>>& insertions, const vector<V> *q, const PagerankOptions<V>& o) {
  K    N    = yt.order();  if (N==0) return {};
  auto ks   = vertexKeys(yt);
  auto vaff = compressContainer(y, pagerankAffectedFrontier(x, y, deletions, insertions), ks);
  auto fa   = [&](K u) { return vaff[u]==true; };
  auto fp   = [&](K u) { y.forEachEdgeKey(u, [&](auto v) { vaff[v] = true; }); };
  return pagerankOmp<ASYNC>(yt, q, o, ks, 0, N, pagerankBasicOmpLoop<ASYNC, DEAD, K, V, decltype(fa), decltype(fp)>, fa, fp);
}
#endif
