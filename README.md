Performance of static vs dynamic [OpenMP]-based [PageRank algorithm] for [link analysis].

Dynamic graphs, which change with time, have many applications. Computing ranks
of vertices from scratch on every update (*static PageRank*) may not be good
enough for an *interactive system*. In such cases, we only want to process ranks
of vertices which are likely to have changed. To handle any new vertices
added/removed, we first *adjust* the *previous ranks* (before the graph
update/batch) with a *scaled 1/N-fill* approach [(1)]. Then, with **naive**
**dynamic approach** we simply run the PageRank algorithm with the *initial ranks*
set to the adjusted ranks. Alternatively, with the (fully) **dynamic approach**
we first obtain a *subset of vertices* in the graph which are likely to be
affected by the update (using BFS/DFS from changed vertices), and then perform
PageRank computation on *only* this *subset of vertices*.

In this experiment, we compare the performance of **static**, **naive dynamic**,
and (fully) **dynamic OpenMP-based PageRank** (along with similar *sequential*
*approaches*). We take *temporal graphs* as input, and add edges to our in-memory
graph in batches of size `10^2 to 10^6`. However we do *not* perform this on
every point on the temporal graph, but *only* on *5 time samples* of the graph
(5 samples are good enough to obtain an average). At each time sample we load
`B` edges (where *B* is the batch size), and perform *static*, *naive dynamic*,
and *dynamic* PageRank. At each time sample, each approach is performed *5*
*times* to obtain an average time for that sample.  A *schedule* of `dynamic, 2048`
is used for *OpenMP-based PageRank* as obtained in [(2)]. We use the
follwing PageRank parameters: damping factor `α = 0.85`, tolerance `τ = 10^-6`,
and limit the maximum number of iterations to `L = 500.` The error between the
current and the previous iteration is obtained with *L1-norm*, and is used to
detect convergence. *Dead ends* in the graph are handled by always teleporting
any vertex in the graph at random (*teleport* approach [(3)]). Error in ranks
obtained for each approach is measured relative to the *sequential static
approach* using *L1-norm*.

From the results, we observe that **naive dynamic and dynamic PageRank are**
**significantly faster than static PageRank for small batch sizes**. However **as**
**the batch size increases**, the **gap** between static and the two dynamic
approaches **decreases** (as one would expect). However, interestingly we note
that there seems to be *little to no difference* between *naive dynamic* and
*dynamic* PageRank. As dynamic PageRank has an added cost of finding the subset
of vertices which might be affected (for which time taken is not considered
here), it seems that **using naive dynamic PageRank is a better option** (which
is also easier to implement).

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
# [751 order; 6952 size; 00000.446 ms; 046 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [751 order; 6952 size; 00000.470 ms; 046 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [751 order; 6952 size; 00000.156 ms; 015 iters.] [4.9179e-06 err.] pagerankSeqNaiveDynamic
# [751 order; 6952 size; 00000.161 ms; 015 iters.] [4.9179e-06 err.] pagerankOmpNaiveDynamic
# [751 order; 6952 size; 00000.162 ms; 015 iters.] [4.9179e-06 err.] pagerankSeqDynamic
# [751 order; 6952 size; 00000.162 ms; 015 iters.] [4.9179e-06 err.] pagerankOmpDynamic
# [802 order; 10532 size; 00000.399 ms; 028 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [802 order; 10532 size; 00000.412 ms; 028 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [802 order; 10532 size; 00000.245 ms; 016 iters.] [2.9494e-06 err.] pagerankSeqNaiveDynamic
# [802 order; 10532 size; 00000.239 ms; 016 iters.] [2.9494e-06 err.] pagerankOmpNaiveDynamic
# [802 order; 10532 size; 00000.249 ms; 016 iters.] [2.9494e-06 err.] pagerankSeqDynamic
# [802 order; 10532 size; 00000.246 ms; 016 iters.] [2.9494e-06 err.] pagerankOmpDynamic
# ...
# [986 order; 24929 size; 00000.746 ms; 023 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [986 order; 24929 size; 00000.719 ms; 023 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [986 order; 24929 size; 00000.356 ms; 011 iters.] [3.1348e-06 err.] pagerankSeqNaiveDynamic
# [986 order; 24929 size; 00000.360 ms; 011 iters.] [3.1348e-06 err.] pagerankOmpNaiveDynamic
# [986 order; 24929 size; 00000.358 ms; 011 iters.] [3.1348e-06 err.] pagerankSeqDynamic
# [986 order; 24929 size; 00000.346 ms; 011 iters.] [3.1348e-06 err.] pagerankOmpDynamic
#
# # Batch size 1e+03
# [751 order; 6952 size; 00000.456 ms; 046 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [751 order; 6952 size; 00000.473 ms; 046 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [751 order; 6952 size; 00000.244 ms; 023 iters.] [8.9126e-06 err.] pagerankSeqNaiveDynamic
# [751 order; 6952 size; 00000.243 ms; 023 iters.] [8.9126e-06 err.] pagerankOmpNaiveDynamic
# [751 order; 6952 size; 00000.253 ms; 023 iters.] [8.9126e-06 err.] pagerankSeqDynamic
# [751 order; 6952 size; 00000.236 ms; 023 iters.] [8.9126e-06 err.] pagerankOmpDynamic
# [802 order; 10532 size; 00000.396 ms; 028 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [802 order; 10532 size; 00000.416 ms; 028 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [802 order; 10532 size; 00000.293 ms; 020 iters.] [8.0420e-07 err.] pagerankSeqNaiveDynamic
# [802 order; 10532 size; 00000.297 ms; 020 iters.] [8.0420e-07 err.] pagerankOmpNaiveDynamic
# [802 order; 10532 size; 00000.299 ms; 020 iters.] [8.0420e-07 err.] pagerankSeqDynamic
# [802 order; 10532 size; 00000.295 ms; 020 iters.] [8.0420e-07 err.] pagerankOmpDynamic
# ...
#
# # Batch size 1e+06
# [986 order; 24929 size; 00000.746 ms; 023 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [986 order; 24929 size; 00000.748 ms; 023 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [986 order; 24929 size; 00000.749 ms; 023 iters.] [0.0000e+00 err.] pagerankSeqNaiveDynamic
# [986 order; 24929 size; 00000.743 ms; 023 iters.] [0.0000e+00 err.] pagerankOmpNaiveDynamic
# [986 order; 24929 size; 00000.737 ms; 023 iters.] [0.0000e+00 err.] pagerankSeqDynamic
# [986 order; 24929 size; 00000.710 ms; 023 iters.] [0.0000e+00 err.] pagerankOmpDynamic
#
#
# Using graph /home/subhajit/data/CollegeMsg.txt ...
# OMP_NUM_THREADS=12
# Temporal edges: 59836
#
# # Batch size 1e+02
# [564 order; 2335 size; 00000.207 ms; 043 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [564 order; 2335 size; 00000.194 ms; 043 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [564 order; 2335 size; 00000.236 ms; 023 iters.] [3.2353e-06 err.] pagerankSeqNaiveDynamic
# [564 order; 2335 size; 00000.282 ms; 023 iters.] [3.2353e-06 err.] pagerankOmpNaiveDynamic
# [564 order; 2335 size; 00000.260 ms; 023 iters.] [3.2353e-06 err.] pagerankSeqDynamic
# [564 order; 2335 size; 00000.327 ms; 023 iters.] [3.2353e-06 err.] pagerankOmpDynamic
# [792 order; 4452 size; 00000.364 ms; 048 iters.] [0.0000e+00 err.] pagerankSeqStatic
# [792 order; 4452 size; 00000.374 ms; 048 iters.] [0.0000e+00 err.] pagerankOmpStatic
# [792 order; 4452 size; 00000.178 ms; 023 iters.] [1.0701e-05 err.] pagerankSeqNaiveDynamic
# [792 order; 4452 size; 00000.175 ms; 023 iters.] [1.0701e-05 err.] pagerankOmpNaiveDynamic
# [792 order; 4452 size; 00000.235 ms; 023 iters.] [1.0701e-05 err.] pagerankSeqDynamic
# [792 order; 4452 size; 00000.614 ms; 023 iters.] [1.0701e-05 err.] pagerankOmpDynamic
# ...
```

[![](https://i.imgur.com/n7Qvkqt.png)][sheetp]
[![](https://i.imgur.com/wn8Lthe.png)][sheetp]

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [The PageRank Citation Ranking: Bringing Order to the Web; Larry Page et al. (1998)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [What's the difference between "static" and "dynamic" schedule in OpenMP?](https://stackoverflow.com/a/10852852/1413259)
- [OpenMP Dynamic vs Guided Scheduling](https://stackoverflow.com/a/43047074/1413259)

<br>
<br>


[![](https://i.imgur.com/sO7WDHn.jpg)](https://in.pinterest.com/pin/636837203543731147/)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/531797868.svg)](https://zenodo.org/badge/latestdoi/531797868)


[(1)]: https://gist.github.com/wolfram77/eb7a3b2e44e3c2069e046389b45ead03
[(2)]: https://github.com/puzzlef/pagerank-openmp-adjust-schedule
[(3)]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
[gist]: https://gist.github.com/wolfram77/170158f966f6c18757434dfa5ba0663f
[charts]: https://imgur.com/a/4RzD9uD
[sheets]: https://docs.google.com/spreadsheets/d/1R4orGRDO_8cKxNOhz48euJQaJ8KyWQ7moxdvruOBN8Y/edit?usp=sharing
[sheetp]: https://docs.google.com/spreadsheets/d/e/2PACX-1vTNA8K91pvoCNlXwt6m-N9Mo3GUHU-JeFL6CwrcVOktN1zpYgJt5Z1jJMPt3We5m1cxjrQcfVO3Qrl3/pubhtml
