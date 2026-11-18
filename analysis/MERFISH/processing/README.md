# MERFISH processing

Set the input and output paths at the top of each script, then run:

```bash
python -u 01_merfish_qc.py
python -u 02_twostage_cca_annotation.py
```

## 1. Cell filtering

Supply a MERFISH `.h5ad` containing only wild-type samples, with sample metadata
already in `.obs` (`experiment`, `sample`, `age`, and `sex` for annotation).
Provide raw counts in `.X` or `.layers['counts']` and per-cell blank-probe counts
in `.obsm['blank_genes']`.

Cells must pass all filters:

| Metric | Passing values |
| --- | --- |
| `obs['transcript_count']` | ≥ 20 |
| `obs['n_genes_by_counts']` | ≥ 10 |
| `obs['volume']` | 100–9,000 µm³ |
| `obs['anisotropy']` | ≤ 4.0 |
| `obs['solidity']` | ≥ 0.50 |
| Blank-probe fraction (blank counts / total counts including blanks) | ≤ 0.10 |

The output is `merfish_qc_filtered.h5ad`. The script populates
`.layers['counts']` from `.X` if the layer is absent.

## 2. Cell-type annotation

Annotation uses every cell that passes QC and a single-nucleus RNA reference with
raw counts in `.X` and cell-type labels in `.obs['celltype']`, including
`Hepatocytes`. The script uses shared genes and two rounds of CCA label transfer to distinguish
hepatocytes first, then resolve non-hepatocyte types. Predictions below the
0.80 confidence cutoff are labeled `Unknown`.

- `merfish_annotated.h5ad` contains all cells included in annotation and their predictions.
- `merfish_final_known.h5ad` contains confidently annotated cells and their MERFISH-only embedding.

QC requires Python with Scanpy and NumPy. Annotation also uses SciPy, pandas,
scikit-learn, matplotlib, harmonypy, leidenalg, umap-learn, and ALLCools.
RAPIDS SingleCell is optional for GPU acceleration.
