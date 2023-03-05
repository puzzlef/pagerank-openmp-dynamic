Design of [OpenMP]-based **Dnamic** [PageRank algorithm] for [link analysis].

Dynamic graphs, which change with time, have many applications. Computing ranks
of vertices from scratch on every update (*static PageRank*) may not be good
enough for an *interactive system*. In such cases, we only want to process ranks
of vertices which are likely to have changed. To handle any new vertices
added/removed, we first *adjust* the *previous ranks* (before the graph
update/batch) with a *scaled 1/N-fill* approach [(1)][adjust-ranks]. Then, with
**naive dynamic approach** we simply run the PageRank algorithm with the
*initial ranks* set to the adjusted ranks. Alternatively, with the (fully)
**dynamic approach** we first obtain a *subset of vertices* in the graph which
are likely to be affected by the update (using BFS/DFS from changed vertices),
and then perform PageRank computation on *only* this *subset of vertices*.

The input data used for the experiments given below is available from the
[SuiteSparse Matrix Collection]. These experiments are done with guidance from
[Prof. Kishore Kothapalli], [Prof. Dip Sankar Banerjee], and [Prof. Sathya Peri].

<br>


### Comparision with Ordered approach

**Unordered PageRank** is the *standard* method of calculating PageRank (as given
in the original PageRank paper by Larry Page et al. [(1)][pagerank-original]),
where *two* *rank vectors* are maintained; one denotes the *present* ranks of
vertices, and the other denotes the *previous* ranks. On the contrary, **ordered**
**PageRank** uses only *one rank vector*, denoting the present ranks [(2)][pagerank].
This is similar to barrierless non-blocking PageRank implementations by
Hemalatha Eedi et al. [(3)][barrierfrees]. As ranks are updated in the same
vector (with each iteration), the order in which ranks of vertices are
calculated *affects* the final result (hence the modifier *ordered*). However,
PageRank is an iteratively converging algorithm, and thus ranks obtained with
either approach are *mostly identical*.

In this experiment ([approach-ordered]), we measure the performance of
**static**, **naive dynamic**, and (fully) **dynamic OpenMP-based ordered**
**PageRank** (along with similar *sequential approaches*). We take *temporal*
*graphs* as input, and add edges to our in-memory graph in batches of size
`10^2` to `10^6`. However we do *not* perform this on every point on the
temporal graph, but *only* on *5 time* *samples* of the graph (5 samples are
good enough to obtain an average). At each time sample we load `B` edges (where
*B* is the batch size), and perform *static*, *naive dynamic*, and *dynamic*
ordered PageRank. At each time sample, each approach is performed *5 times* to
obtain an average time for that sample.  A *schedule* of `dynamic, 2048` is used
for *OpenMP-based PageRank* as obtained in [(5)][adjust-schedule]. We use the
follwing PageRank parameters: damping factor `α = 0.85`, tolerance `τ = 10^-6`,
and limit the maximum number of iterations to `L = 500.` The error between the
present and the previous iteration is obtained with *L1-norm*, and is used to
detect convergence. *Dead ends* in the graph are dealt with by always
teleporting any vertex in the graph at random (*teleport* approach
[(6)][dead-ends]). Error in ranks obtained for each approach is measured
relative to the *sequential static approach* using *L1-norm*.

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

[approach-ordered]: https://github.com/puzzlef/pagerank-openmp-dynamic-tofu/tree/approach-ordered

<br>


### Comparision with Static approach

In this experiment ([compare-static]), we compare the performance of **static**,
**naive dynamic**, and (fully) **dynamic OpenMP-based PageRank** (along with
similar *sequential approaches*). We take *temporal graphs* as input, and add
edges to our in-memory graph in batches of size `10^2 to 10^6`. However we do
*not* perform this on every point on the temporal graph, but *only* on *5 time*
*samples* of the graph (5 samples are good enough to obtain an average). At each
time sample we load `B` edges (where *B* is the batch size), and perform
*static*, *naive dynamic*, and *dynamic* PageRank. At each time sample, each
approach is performed *5* *times* to obtain an average time for that sample.  A
*schedule* of `dynamic, 2048` is used for *OpenMP-based PageRank* as obtained in
[(2)][adjust-schedule]. We use the follwing PageRank parameters: damping factor
`α = 0.85`, tolerance `τ = 10^-6`, and limit the maximum number of iterations to
`L = 500.` The error between the current and the previous iteration is obtained
with *L1-norm*, and is used to detect convergence. *Dead ends* in the graph are
handled by always teleporting any vertex in the graph at random (*teleport*
approach [(3)][dead-ends]). Error in ranks obtained for each approach is
measured relative to the *sequential static approach* using *L1-norm*.

From the results, we observe that **naive dynamic and dynamic PageRank are**
**significantly faster than static PageRank for small batch sizes**. However **as**
**the batch size increases**, the **gap** between static and the two dynamic
approaches **decreases** (as one would expect). However, interestingly we note
that there seems to be *little to no difference* between *naive dynamic* and
*dynamic* PageRank. As dynamic PageRank has an added cost of finding the subset
of vertices which might be affected (for which time taken is not considered
here), it seems that **using naive dynamic PageRank is a better option** (which
is also easier to implement). All outputs are saved in a [gist]. Some [charts]
are also included below, generated from [sheets].

[![](https://i.imgur.com/n7Qvkqt.png)][sheetp]
[![](https://i.imgur.com/wn8Lthe.png)][sheetp]

[compare-static]: https://github.com/puzzlef/pagerank-openmp-dynamic-tofu/tree/compare-static

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


[pagerank-original]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427
[pagerank]: https://github.com/puzzlef/pagerank
[barrierfrees]: https://ieeexplore.ieee.org/document/9407114
[adjust-ranks]: https://gist.github.com/wolfram77/eb7a3b2e44e3c2069e046389b45ead03
[adjust-schedule]: https://github.com/puzzlef/pagerank-openmp
[dead-ends]: https://gist.github.com/wolfram77/94c38b9cfbf0c855e5f42fa24a8602fc
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
