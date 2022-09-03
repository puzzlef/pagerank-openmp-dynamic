Performance of static vs dynamic [OpenMP]-based ordered [PageRank algorithm]
for [link analysis].

**Unordered PageRank** is the *standard* method of calculating PageRank (as
given in the original PageRank paper by Larry Page et al. [(1)]), where *two*
*rank vectors* are maintained; one denotes the *present* ranks of vertices, and
the other denotes the *previous* ranks. On the contrary, **ordered PageRank**
uses only *one rank vector*, denoting the present ranks [(2)]. This is similar
to barrierless non-blocking PageRank implementations by Hemalatha Eedi et al.
[(3)]. As ranks are updated in the same vector (with each iteration), the order
in which ranks of vertices are calculated *affects* the final result (hence the
modifier *ordered*). However, PageRank is an iteratively converging algorithm,
and thus ranks obtained with either approach are *mostly identical*.

Dynamic graphs change with time, and have many applications. Computing PageRank
from scratch on every update (*static PageRank*) might not be suitable for
*interactive systems*. In that case, we only want to recompute ranks of vertices
which are likely to have altered. To deal with any fresh vertices added/removed,
we first *tailor* the *previous ranks* (before the graph update/batch) with a
*scaled 1/N-fill* approach [(4)]. Then, with **naive dynamic approach** we
simply re-run the PageRank algorithm with the *initial ranks* set to the
tailored ranks. Alternately, with the (fully) **dynamic approach** we first
obtain a *portion of vertices* in the graph which are likely to be affected by
the update (using BFS/DFS from changed vertices), and then perform PageRank
computation on *only* this *portion of vertices*.

In this experiment, we measure the performance of **static**, **naive dynamic**,
and (fully) **dynamic OpenMP-based ordered PageRank** (along with similar
*sequential approaches*). We take *temporal graphs* as input, and add edges to
our in-memory graph in batches of size `10^2 to 10^6`. However we do *not*
perform this on every point on the temporal graph, but *only* on *5 time*
*samples* of the graph (5 samples are good enough to obtain an average). At each
time sample we load `B` edges (where *B* is the batch size), and perform
*static*, *naive dynamic*, and *dynamic* ordered PageRank. At each time sample,
each approach is performed *5 times* to obtain an average time for that
sample.  A *schedule* of `dynamic, 2048` is used for *OpenMP-based PageRank* as
obtained in [(5)]. We use the follwing PageRank parameters: damping factor
`α = 0.85`, tolerance `τ = 10^-6`, and limit the maximum number of iterations to
`L = 500.` The error between the present and the previous iteration is obtained with
*L1-norm*, and is used to detect convergence. *Dead ends* in the graph are dealt
with by always teleporting any vertex in the graph at random (*teleport*
approach [(6)]). Error in ranks obtained for each approach is measured relative
to the *sequential static approach* using *L1-norm*.

From the results, we observe that **naive dynamic and dynamic PageRank are**
**significantly faster than static PageRank for small batch sizes**. However
**as** **the batch size increases**, the **separation** between static and the
two dynamic approaches **reduce** (as you would expect). Interestingly we note
that (again) there seems to be *little to no difference* between *naive dynamic*
and *dynamic* PageRank. As dynamic PageRank has an added cost of finding the
subset of vertices which might be affected (for which time taken is not
considered here), it seems that **using naive dynamic PageRank is a better**
**option** (which is also easier to implement). We also observe that the
iterations required for OpenMP-based approaches is slightly higher than the
sequential approaches. This may be due to the multithreaded OpenMP-based
approaches becoming closer in behavior to an unordered approach (due to parallel
threads).

All outputs are saved in a [gist] and a small part of the output is listed here.
Some [charts] are also included below, generated from [sheets]. The input data
used for this experiment is available from the [SuiteSparse Matrix Collection].
This experiment was done with guidance from [Prof. Kishore Kothapalli],
[Prof. Dip Sankar Banerjee], and [Prof. Sathya Peri].

<br>

```bash
$ g++ -std=c++17 -O3 -fopenmp main.cxx
$ ./a.out ~/data/email-Eu-core-temporal.txt
$ ./a.out ~/data/CollegeMsg.txt
$ ...

# Using graph /home/subhajit/data/email-Eu-core-temporal.txt ...
# OMP_NUM_THREADS=12
# Temporal edges: 332335
#
# # Batch size 1e+02
# [751 order; 6952 size; 00000.385 ms; 036 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [751 order; 6952 size; 00000.439 ms; 036 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [751 order; 6952 size; 00000.145 ms; 013 iters.] [4.7024e-06 err.] pagerankSeqNaiveDynamic
# [751 order; 6952 size; 00000.168 ms; 013 iters.] [4.7024e-06 err.] pagerankOmpNaiveDynamic
# [751 order; 6952 size; 00000.145 ms; 013 iters.] [4.7024e-06 err.] pagerankSeqDynamic
# [751 order; 6952 size; 00000.168 ms; 013 iters.] [4.7024e-06 err.] pagerankOmpDynamic
# [802 order; 10532 size; 00000.560 ms; 036 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [802 order; 10532 size; 00000.595 ms; 036 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [802 order; 10532 size; 00000.168 ms; 010 iters.] [1.0612e-06 err.] pagerankSeqNaiveDynamic
# [802 order; 10532 size; 00000.183 ms; 010 iters.] [1.0612e-06 err.] pagerankOmpNaiveDynamic
# [802 order; 10532 size; 00000.163 ms; 010 iters.] [1.0612e-06 err.] pagerankSeqDynamic
# [802 order; 10532 size; 00000.179 ms; 010 iters.] [1.0612e-06 err.] pagerankOmpDynamic
# ...
# [986 order; 24929 size; 00001.264 ms; 036 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [986 order; 24929 size; 00001.302 ms; 036 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [986 order; 24929 size; 00000.260 ms; 007 iters.] [1.4549e-06 err.] pagerankSeqNaiveDynamic
# [986 order; 24929 size; 00000.271 ms; 007 iters.] [1.4549e-06 err.] pagerankOmpNaiveDynamic
# [986 order; 24929 size; 00000.257 ms; 007 iters.] [1.4549e-06 err.] pagerankSeqDynamic
# [986 order; 24929 size; 00000.261 ms; 007 iters.] [1.4549e-06 err.] pagerankOmpDynamic
#
# # Batch size 1e+03
# [751 order; 6952 size; 00000.396 ms; 036 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [751 order; 6952 size; 00000.460 ms; 036 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [751 order; 6952 size; 00000.221 ms; 019 iters.] [6.6651e-07 err.] pagerankSeqNaiveDynamic
# [751 order; 6952 size; 00000.246 ms; 019 iters.] [6.6651e-07 err.] pagerankOmpNaiveDynamic
# [751 order; 6952 size; 00000.221 ms; 019 iters.] [6.6651e-07 err.] pagerankSeqDynamic
# [751 order; 6952 size; 00000.254 ms; 019 iters.] [6.6651e-07 err.] pagerankOmpDynamic
# [802 order; 10532 size; 00000.568 ms; 036 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [802 order; 10532 size; 00000.632 ms; 036 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [802 order; 10532 size; 00000.354 ms; 021 iters.] [2.1089e-07 err.] pagerankSeqNaiveDynamic
# [802 order; 10532 size; 00000.377 ms; 021 iters.] [2.1089e-07 err.] pagerankOmpNaiveDynamic
# [802 order; 10532 size; 00000.336 ms; 021 iters.] [2.1089e-07 err.] pagerankSeqDynamic
# [802 order; 10532 size; 00000.357 ms; 021 iters.] [2.1089e-07 err.] pagerankOmpDynamic
# ...
#
#
# Using graph /home/subhajit/data/CollegeMsg.txt ...
# OMP_NUM_THREADS=12
# Temporal edges: 59836
#
# # Batch size 1e+02
# [564 order; 2335 size; 00000.194 ms; 043 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [564 order; 2335 size; 00000.211 ms; 043 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [564 order; 2335 size; 00000.085 ms; 018 iters.] [9.2026e-07 err.] pagerankSeqNaiveDynamic
# [564 order; 2335 size; 00000.093 ms; 018 iters.] [9.2026e-07 err.] pagerankOmpNaiveDynamic
# [564 order; 2335 size; 00000.085 ms; 018 iters.] [9.2026e-07 err.] pagerankSeqDynamic
# [564 order; 2335 size; 00000.101 ms; 018 iters.] [9.2026e-07 err.] pagerankOmpDynamic
# [792 order; 4452 size; 00000.314 ms; 041 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [792 order; 4452 size; 00000.403 ms; 041 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [792 order; 4452 size; 00000.201 ms; 025 iters.] [4.8333e-07 err.] pagerankSeqNaiveDynamic
# [792 order; 4452 size; 00000.247 ms; 025 iters.] [4.8333e-07 err.] pagerankOmpNaiveDynamic
# [792 order; 4452 size; 00000.204 ms; 025 iters.] [4.8333e-07 err.] pagerankSeqDynamic
# [792 order; 4452 size; 00000.247 ms; 025 iters.] [4.8333e-07 err.] pagerankOmpDynamic
# ...
```

[![](https://i.imgur.com/CitMI5C.png)][sheetp]
[![](https://i.imgur.com/tJ6G5aP.png)][sheetp]

<br>
<br>


## References

- [An Efficient Practical Non-Blocking PageRank Algorithm for Large Scale Graphs; Hemalatha Eedi et al. (2021)](https://ieeexplore.ieee.org/document/9407114)
- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [The PageRank Citation Ranking: Bringing Order to the Web; Larry Page et al. (1998)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [What's the difference between "static" and "dynamic" schedule in OpenMP?](https://stackoverflow.com/a/10852852/1413259)
- [OpenMP Dynamic vs Guided Scheduling](https://stackoverflow.com/a/43047074/1413259)

<br>
<br>


[![](https://i.imgur.com/sO7WDHn.jpg)](https://in.pinterest.com/pin/636837203543731147/)<br>
[![DOI](https://zenodo.org/badge/532019117.svg)](https://zenodo.org/badge/latestdoi/532019117)


[(1)]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[(2)]: https://github.com/puzzlef/pagerank-ordered-vs-unordered
[(3)]: https://ieeexplore.ieee.org/document/9407114
[(4)]: https://gist.github.com/wolfram77/eb7a3b2e44e3c2069e046389b45ead03
[(5)]: https://github.com/puzzlef/pagerank-openmp-adjust-schedule
[(6)]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
[gist]: https://gist.github.com/wolfram77/f8f850295c17ea3e6907e5e497574baa
[charts]: https://imgur.com/a/h8jyG2d
[sheets]: https://docs.google.com/spreadsheets/d/1JwEHW6P4L9OKXMcAbc_jfPr5YT2Cog13NWcRxTX3Fe4/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vS-5sbMaxLB7H4JyzsXIINm8nFLbLRap-IZDxaKfvuHLUicEc-6rdm-Le9qhzsnoYUTMLlkMRTx0oDw/pubhtml
