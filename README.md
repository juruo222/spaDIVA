# spaDIVA

spaDIVA is a graph-based generative framework for disentangling two paired spatial omics layers into shared and modality-specific representations. It also produces an integrated embedding for spatial-domain identification and downstream analysis.

## What spaDIVA learns

- Shared and modality-specific latent representations.
- An integrated representation for visualization and clustering.
- Unique factor ratio: a feature-level measure of modality-specific dependence beyond the shared representation.

In the tutorials, `Z` is the WNN shared embedding and `Z_poe` is a saved PoE sample; single-section tutorials also store the integrated embedding as `Z_W`. Interpretation utilities use `Z_poe` by default, fall back to `Z` if needed, and accept `use="Z"` explicitly.

## Installation

Run from the cloned repository root:

```bash
conda env create -f environment.yml
conda activate spaDIVA
python -m pip install -e . --no-deps
```

If this `spaDIVA` environment is already installed, start with `conda activate spaDIVA`.

The [environment file](environment.yml) targets Linux and Windows on x86_64 and includes CUDA-enabled PyTorch, HNSW matching, and mclust clustering (R 4.4, rpy2 3.6, mclust 6.1). The editable install registers this checkout without reinstalling dependencies. Tutorials default to CPU (`USE_CUDA = False`); use `True` with a compatible NVIDIA GPU and driver. Clustering uses mclust's EEE model.

## Tutorials and data

- [Simulation quickstart](tutorials/tutorial_simulation.ipynb): uses the included [example data](examples/simulation_data/README.md).
- [Unique factor ratio](tutorials/tutorial_unique_factor_ratio_simulation.ipynb): a self-contained simulation example; no external-model environment is required.
- [P22 ATAC-RNA](tutorials/tutorial_P22_ATAC_RNA.ipynb) and [lymph node RNA-ADT](tutorials/tutorial_human_lymph_node_RNA_ADT.ipynb).
- [E13/E15/E18](tutorials/tutorial_E13E15E18_multi_slice.ipynb) and [P21/P22](tutorials/tutorial_P21P22_multi_slice.ipynb): joint multi-section analysis.

Start Jupyter from the cloned repository root. For the full datasets, extract the spaDIVA data package and set its root first:

Linux (Bash):

```bash
export SPADIVA_DATA_ROOT="/path/to/spaDIVA_data"
jupyter lab
```

Windows (PowerShell):

```powershell
$env:SPADIVA_DATA_ROOT = "D:\path\to\spaDIVA_data"
jupyter lab
```

The selected directory must contain `datasets/`. These tutorials demonstrate the software workflow; the manuscript's frozen results and external MultiVI/totalVI analyses are available in the [spaDIVA reproducibility repository](https://github.com/juruo222/spaDIVA_reproducibility).
