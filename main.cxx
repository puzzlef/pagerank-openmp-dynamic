#include <utility>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>
#include "src/main.hxx"

using namespace std;




// You can define datatype with -DTYPE=...
#ifndef TYPE
#define TYPE float
#endif
// You can define number of threads with -DMAX_THREADS=...
#ifndef MAX_THREADS
#define MAX_THREADS 12
#endif




void runPagerankBatch(const string& data, size_t batch, size_t skip, int repeat) {
  using K = int;
  using T = TYPE;
  enum NormFunction { L0=0, L1=1, L2=2, Li=3 };
  vector<T> ranksOld, ranksAdj;
  vector<T> *initStatic  = nullptr;
  vector<T> *initDynamic = &ranksAdj;

  OutDiGraph<K> x;
  stringstream  stream(data);
  while (true) {
    // Lets skip some edges.
    if (!readSnapTemporalW(x, stream, skip)) break;
    auto xt  = transposeWithDegree(x);
    auto a0  = pagerankMonolithicSeq<true>(x, xt, initStatic);
    auto ksOld = vertexKeys(x);
    ranksOld   = a0.ranks;

    // Read batch to be processed.
    auto y  = duplicate(x);
    if (!readSnapTemporalW(y, stream, batch)) break;
    auto ks = vertexKeys(y);
    auto yt = transposeWithDegree(y);

    // Find ordered pagerank using a single thread (static).
    auto a1 = pagerankMonolithicSeq<true>(y, yt, initStatic, {repeat});
    auto e1 = l1Norm(a1.ranks, a1.ranks);
    printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankSeqStatic\n", y.order(), y.size(), a1.time, a1.iterations, e1);

    // Find ordered pagerank accelerated with OpenMP (static).
    auto a2 = pagerankMonolithicOmp<true>(y, yt, initStatic, {repeat});
    auto e2 = l1Norm(a2.ranks, a1.ranks);
    printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpStatic\n", y.order(), y.size(), a2.time, a2.iterations, e2);

    // Adjust ranks for dynamic Pagerank.
    ranksAdj.resize(y.span());
    adjustRanks(ranksAdj, ranksOld, ksOld, ks, 0.0f, float(ksOld.size())/ks.size(), 1.0f/ks.size());

    // Find ordered pagerank using a single thread (naive dynamic).
    auto a3 = pagerankMonolithicSeq<true>(y, yt, initDynamic, {repeat});
    auto e3 = l1Norm(a3.ranks, a1.ranks);
    printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankSeqNaiveDynamic\n", y.order(), y.size(), a3.time, a3.iterations, e3);

    // Find ordered pagerank accelerated with OpenMP (naive dynamic).
    auto a4 = pagerankMonolithicOmp<true>(y, yt, initDynamic, {repeat});
    auto e4 = l1Norm(a4.ranks, a1.ranks);
    printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpNaiveDynamic\n", y.order(), y.size(), a4.time, a4.iterations, e4);

    // Find ordered pagerank using a single thread (dynamic).
    auto a5 = pagerankMonolithicSeqDynamic<true>(x, xt, y, yt, initDynamic, {repeat});
    auto e5 = l1Norm(a5.ranks, a1.ranks);
    printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankSeqDynamic\n", y.order(), y.size(), a5.time, a5.iterations, e5);

    // Find ordered pagerank accelerated with OpenMP (dynamic).
    auto a6 = pagerankMonolithicOmpDynamic<true>(x, xt, y, yt, initDynamic, {repeat});
    auto e6 = l1Norm(a6.ranks, a1.ranks);
    printf("[%zu order; %zu size; %09.3f ms; %03d iters.] [%.4e err.] pagerankOmpDynamic\n", y.order(), y.size(), a6.time, a6.iterations, e6);

    // Now time to move on to next batch.
    x = move(y);
  }
}


void runPagerank(const string& data, int repeat) {
  size_t M = countLines(data), steps = 10;
  printf("Temporal edges: %zu\n\n", M);
  for (size_t batch=100; batch<=1000000; batch*=10) {
    size_t skip = max(int64_t(M/steps) - int64_t(batch), 0L);
    printf("# Batch size %.0e\n", double(batch));
    runPagerankBatch(data, batch, skip, repeat);
    printf("\n");
  }
}


int main(int argc, char **argv) {
  char *file = argv[1];
  int repeat = argc>2? stoi(argv[2]) : 5;
  printf("Using graph %s ...\n", file);
  string data = readFile(file);
  omp_set_num_threads(MAX_THREADS);
  printf("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  runPagerank(data, repeat);
  printf("\n");
  return 0;
}
