# Voronoi-Geometric-K-Server

We present a heuristic algorithm for the online k-Server problem. We
compute the expected minimum distance with respect to the k points
using a simplicial decomposition of the Voronoi diagram.

We greedily assign servers based on both the distance and the
resulting expected minimum distance. In theory this should prevent
many of the usual adversary tricks that cause unbounded competitive
ratios.

