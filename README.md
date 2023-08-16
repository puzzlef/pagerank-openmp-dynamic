Design of [OpenMP]-based **Dynamic** [PageRank algorithm] for [link analysis].

I have tried using **contribution vectors** `r/deg` instead of *rank vectors* `r`. Using *contribution vectors* requires `|E|` memory accesses per iteration compared to using *rank vectors* (`2|E|` accesses). However, it involves calculating *contributions vector* from intial ranks (pre-processing), calculating *rank vector* from contributions (post-processing), and measuring rank change per vertex (additional `|V|` memory accesses). For large batch updates (as well as static PageRank), using contributions vector could improve performance. In addition, if we track contributions with dynamic PageRank, both pre-processing and post-processing cost could be neglected.

Using contributions vector appears to provide `~15%` speedup with Static/Naive-dynamic PageRank, and `> 10%` speedup with Dynamic Frontier PageRank after a batch size of `10^-4|E|`.

> See
> [code](https://github.com/puzzlef/pagerank-openmp-dynamic/tree/approach-contrib),
> [output](https://gist.github.com/wolfram77/fd9c5ce2e42913f49bb62dbeb195bddc), or
> [sheets].

[![](https://i.imgur.com/Xo7oxFO.png)][sheets]
[![](https://i.imgur.com/uBT2eoC.png)][sheets]

[sheets]: https://docs.google.com/spreadsheets/d/1_vCSYdZsnpkoIQpQ9Jtnb7VCfntbSsptYHpT4COLVkI/edit?usp=sharing

<br>
<br>


## References

- [PageRank Algorithm, Mining massive Datasets (CS246), Stanford University](https://www.youtube.com/watch?v=ke9g8hB0MEo)
- [The PageRank Citation Ranking: Bringing Order to the Web; Larry Page et al. (1998)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427)
- [The University of Florida Sparse Matrix Collection; Timothy A. Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [What's the difference between "static" and "dynamic" schedule in OpenMP?](https://stackoverflow.com/a/10852852/1413259)

<br>
<br>


[![](https://img.youtube.com/vi/BSuyYQAI_3g/maxresdefault.jpg)](https://www.youtube.com/watch?v=BSuyYQAI_3g)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)


[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[OpenMP]: https://en.wikipedia.org/wiki/OpenMP
[PageRank algorithm]: https://en.wikipedia.org/wiki/PageRank
[link analysis]: https://en.wikipedia.org/wiki/Network_theory#Link_analysis
