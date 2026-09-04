# Simulated spatial multi-omics data

This folder contains the simulated input data used by `tutorials/tutorial_simulation.ipynb`.

Files:

- `ADT_100.h5ad`: simulated omics 1 data with spatial coordinates and category labels.
- `RNA_ZINB.h5ad`: simulated omics 2 data with spatial coordinates and category labels.

The simulation includes shared and modality-specific spatial factors.
Both files store counts in `layers["counts"]` and `log1p(counts)` in `X`;
these are different preprocessing starting points.
