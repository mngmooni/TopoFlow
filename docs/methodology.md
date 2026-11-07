# Methodology overview

Core chart: conservative segment bands → shared waypoint intervals → contracted `tanh` decoding → geometric soundness for successfully constructed charts.

Topology: binary above/below codes specify intended routes in an x-monotone family. Valid-cut assumptions are required before interpreting the reduced crossing word as a homotopy certificate.

Generative learning: straight-interpolation conditional Flow Matching in encoded chart coordinates. Reference encoding uses a contracted width and clipping; the target is the **encoded empirical distribution**, not necessarily the exact inverse of the raw reference-path distribution.

The full mathematical statements and supporting claims are in the manuscript and `results/final/CHART_SOUNDNESS.md`.
