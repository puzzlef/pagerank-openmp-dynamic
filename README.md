Design of OpenMP-based Parallel Dynamic [PageRank algorithm] for measuring importance.

PageRank serves as an algorithm assessing the significance of nodes within a network through the assignment of numerical scores based on link structure. Its applications span web page ranking, identification of misinformation, traffic flow prediction, and protein target identification. The growing availability of extensive graph-based data has spurred interest in parallel algorithms for PageRank computation.

In real-world scenarios, graphs often evolve over time, with frequent edge insertions and deletions rendering recomputation of PageRank from scratch impractical, especially for small, rapid changes. Existing strategies optimize by iterating from the previous snapshot's ranks, reducing the required iterations for convergence. To further enhance efficiency, it becomes crucial to recalibrate the ranks of only those vertices likely to undergo changes. A common approach entails identifying reachable vertices from the updated graph regions and limiting processing to such vertices. However, if updates are randomly distributed, they may frequently fall within dense graph regions, necessitating the processing of a substantial portion of the graph.

To mitigate computational effort, one can incrementally expand the set of affected vertices from the updated graph region, rather than processing all reachable vertices from the initial iteration. Moreover, it is possible to skip processing a vertex's neighbors if the change in its rank is small and expected to have minimal impact on the ranks of neighboring vertices. Here, we introduce the Dynamic Frontier (DF) anf Dynamic Frontier with Pruning (DF-P) approaches for Updating PageRank, which addresses these considerations.

<br>


Below we plot the average time taken by Static, Naive-dynamic (ND), Dynamic Traversal (DT), our improved Dynamic Frontier (DF), and Dynamic Frontier with Pruning (DF-P) PageRank on 5 real-world dynamic graphs, with batch updates of size `10^-5|Eᴛ|` to `10^-3|Eᴛ|`. The labels indicate the speedup of each approach with respect to Static PageRank. DF PageRank is, on average, `8.0×`, `4.5×`, and `3.2×` faster than Static PageRank, and is, on average, `1.3×`, `1.1×`, and `1.5×` faster than DT PageRank, a widely used approach for updating PageRank on dynamic graphs. In contrast, DF-P PageRank is, on average, `26.2×`, `11.9×`, and `7.5×` faster than Static PageRank and, on average, `4.2×`, `2.8×`, and `3.6×` faster than DT PageRank on identical batch updates.

[![](https://i.imgur.com/YdjQWfH.png)][sheets-o1]

Next, we plot the Error comparison of Static, ND, DT, DF, and DF-P PageRank with respect to a Reference Static PageRank (with a tolerance `𝜏` of `10^−100` and limited to `500` iterations), using L1-norm.

[![](https://i.imgur.com/h2ZErIn.png)][sheets-o1]

Finally, we plot the strong scaling behaviour of DF and DF-P PageRank. With doubling of threads, DF PageRank exhibits an average performance scaling of `1.8×`, while DF-P PageRank exhibits an average performance scaling of `1.7×`.

[![](https://i.imgur.com/uahK7bg.png)][sheets-o2]

Refer to our technical reports for more details: \
[An Incrementally Expanding Approach for Updating PageRank on Dynamic Graphs][report1]. \
[DF* PageRank: Improved Incrementally Expanding Approaches for Updating PageRank on Dynamic Graphs][report2].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.

[PageRank algorithm]: https://www.cis.upenn.edu/~mkearns/teaching/NetworkedLife/pagerank.pdf
[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[Prof. Sathya Peri]: https://people.iith.ac.in/sathya_p/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
[sheets-o1]: https://docs.google.com/spreadsheets/d/1gAPAmS6mLoZ2VqhUp0Y60BSZW-IR-SxaDLnsfRJqwig/edit?usp=sharing
[sheets-o2]: https://docs.google.com/spreadsheets/d/1S1Iciq3z3rKoBb4gY_oyOw8RMB0-2Z7vD3-jKus4bx8/edit?usp=sharing
[report1]: https://arxiv.org/abs/2401.03256
[report2]: https://arxiv.org/abs/2401.15870

<br>
<br>


### Code structure

The code structure of Dynamic Frontier (DF), and Dynamic Frontier with Pruning (DF-P) PageRank is as follows:

```bash
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
- inc/batch.hxx: Batch update generation functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/Graph.hxx: Graph data structure functions
- inc/main.hxx: Main header
- inc/mtx.hxx: Graph file reading functions
- inc/pagerank.hxx: PageRank algorithms
- inc/pagerankPrune.hxx: Dynamic Frontier with Pruning PageRank algorithms
- inc/properties.hxx: Graph Property functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/symmetricize.hxx: Graph Symmetricization functions
- inc/transpose.hxx: Graph transpose functions
- inc/update.hxx: Update functions
- main.cxx: Experimentation code
- process.js: Node.js script for processing output logs
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

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


[![](https://i.imgur.com/ol8RPAQ.jpg)](https://www.youtube.com/watch?v=yqO7wVBTuLw&pp)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
[![DOI](https://zenodo.org/badge/531797868.svg)](https://zenodo.org/doi/10.5281/zenodo.7044645)
