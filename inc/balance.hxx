#pragma once
#include <vector>
#include "_main.hxx"

using std::vector;




/**
 * Partition the vertices of a graph, such that the edges are balanced.
 * @param a partition boundaries
 * @param x original graph
 * @param partitions number of partitions
 */
template <class G, class K>
inline void edgeBalanceW(vector<K>& a, const G& x, size_t partitions) {
  size_t S = x.span();
  size_t N = x.order();
  size_t M = x.size();
  size_t P = ceilDiv(N, partitions), p = 0;
  a.clear();
  a.push_back(0);
  for (K v=0; v<S; ++v) {
    p += x.degree(v);
    if (p >= P) {
      a.push_back(v+1);
      p = 0;
    }
  }
  a.push_back(S);
}
