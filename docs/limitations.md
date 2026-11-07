# Limitations

- Static 2D, x-monotone, polygonal paths; no dynamic or tracking guarantee.
- Conservative chart does not parameterize the unrestricted feasible path space.
- Trained fixed-resolution model differs from adaptive chart-construction coverage.
- Finite sampling-planner references do not characterize every feasible trajectory.
- Full model reproducibility needs dataset regeneration and checkpoint retraining.
- Reported baseline/latency comparison is implementation- and hardware-specific.
- Encoded reference clipping is a known diagnostic discrepancy and its causal relationship to boundary hugging is unestablished.
