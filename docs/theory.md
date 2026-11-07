# Geometric and topological claims

**Geometric soundness:** Under strictly increasing horizontal grid, conservative per-segment obstacle bounds, nondegenerate waypoint intersections, valid endpoints, and finite latent coordinates, each decoded polygonal segment lies in a feasible rectangular band. This is a segment-level, not just waypoint-level, guarantee.

**Homotopy consistency:** Two decoded paths from the same chart deform continuously into each other through convex waypoint interpolation.

**Route-word correspondence:** Additional conditions on obstacle cuts and endpoint-relative geometry are needed to identify the common class with the intended reduced crossing word. Stress-test layouts with invalid cuts do not receive this certificate.

**Coverage:** Construction of a refined chart is not equivalent to generation by a neural network trained on 60 coordinates.
