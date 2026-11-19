# MERFISH senescent-cell mapping and SenSpots

Set the paths at the top of each script. Use the same `SECTION_DIR` in all three
scripts and the same `OUT_DIR` in the mapping and SenSpot scripts.

```bash
python -u 00_export_sections.py
python -u 01_map_snc.py
python -u 02_find_senspots.py
```

The last two scripts process all exported sections by default. Pass `--sid <section>`
to either script to process a single section. If you change `--top-n` for mapping,
pass the same value to SenSpot detection.

## Inputs

- `MERFISH_H5AD` is `merfish_final_known.h5ad` from [MERFISH processing](../processing/).
  It needs log-normalized expression in `.layers['lognorm']`, spatial coordinates
  in µm in `.obsm['spatial']`, and `sample`, `age`, `sex`, and `celltype_final`
  in `.obs`.
- `REFERENCE_H5AD` is a single-nucleus RNA reference with raw counts in `.X`
  and `celltype`, `age`, and `sex` in `.obs`. These values must match the
  corresponding MERFISH annotations.
- `SNC_META_CSV` has reference cell IDs as its first column and
  `predicted_senescence_label` values of `senescent` or `non-senescent`.
  Each reference group used for transfer needs both labels.

## Workflow and outputs

1. **Export sections.** Write one `{sid}.h5ad` per section and a `sections.tsv`
   manifest to `SECTION_DIR`.
2. **Map senescent cells.** Within each section and cell type, match reference
   nuclei by age and sex. Use shared genes from the [SDTMC signature](SDTMC_147_genes.txt)
   to link each reference nucleus to up to 300 MERFISH cells by cosine similarity
   (minimum 0.2). Cells with at least three incoming links from labeled nuclei
   are ranked by the fraction of link weight from senescent nuclei. The highest-ranked
   cells are called senescent in proportion to the matched reference prevalence.
   Per-cell labels and scores are saved to `OUT_DIR/labels/snc_labels_{sid}.csv.gz`.
3. **Find SenSpots.** Cluster senescent cells within each section with DBSCAN
   (50 µm radius, minimum five cells). Assign other cells to the nearest
   clustered senescent cell's SenSpot if the distance is less than 50 µm.
   Membership tables, section plots, and summaries are saved under
   `OUT_DIR/senspots/topn300_eps50/` with the default settings.

The scripts use Python, Scanpy, AnnData, NumPy, pandas, SciPy, scikit-learn, and
Matplotlib. A GPU is optional for the upstream annotation workflow. The mapping
scripts run on CPU. On the manuscript dataset, section export used about 200 GB
of RAM and label transfer used up to about 150 GB per process.
