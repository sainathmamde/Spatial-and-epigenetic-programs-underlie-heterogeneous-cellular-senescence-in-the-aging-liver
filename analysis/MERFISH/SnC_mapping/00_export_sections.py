#!/usr/bin/env python
# ============================================================================================
# Step 0 — split the annotated MERFISH object into one h5ad per tissue section
# --------------------------------------------------------------------------------------------
# SnC mapping and SenSpot detection run per section. Exporting once means each per-section job
# reads ~100 MB instead of the full multi-GB object.
#
# INPUT   MERFISH_H5AD — merfish_final_known.h5ad from the annotation step
#         (01.processing/github_code/02_twostage_cca_annotation.py). Needs:
#           .layers['lognorm'] (or log1p-CP10K in .X), obsm['spatial'] (µm),
#           obs 'sample' (section id), 'age', 'sex', 'celltype_final'
#
# OUTPUT  SECTION_DIR/{sid}.h5ad   one section; X = log1p(CP10K), layers kept,
#                                  obs gains global_x / global_y
#         SECTION_DIR/sections.tsv sid, age, sex, n_cells — read by steps 1 and 2
#
# RUN     python -u 00_export_sections.py [--overwrite]
# ============================================================================================

import os
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

# ════════════════════════════════════════════════════════════════════════════════════════════
#  PATHS  —  edit before running
# ════════════════════════════════════════════════════════════════════════════════════════════
MERFISH_H5AD = '/path/to/twostage_cca_annotation/merfish_final_known.h5ad'
SECTION_DIR  = '/path/to/output/sections'

SECTION_COL = 'sample'      # obs column holding the section id


def main():
    p = argparse.ArgumentParser(description='Split the annotated MERFISH object by section.')
    p.add_argument('--overwrite', action='store_true', help='re-export existing sections')
    args = p.parse_args()

    bad = [n for n, v in [('MERFISH_H5AD', MERFISH_H5AD), ('SECTION_DIR', SECTION_DIR)]
           if v.startswith('/path/to')]
    if bad:
        raise SystemExit(f'ERROR: set the placeholder path(s) at the top of this script: {bad}')
    os.makedirs(SECTION_DIR, exist_ok=True)

    print(f'Reading {MERFISH_H5AD}')
    adata = sc.read_h5ad(MERFISH_H5AD)
    print(f'  {adata.n_obs:,} cells x {adata.n_vars} genes')
    for col in (SECTION_COL, 'age', 'sex', 'celltype_final'):
        if col not in adata.obs:
            raise SystemExit(f"obs has no column {col!r}")
    if 'spatial' not in adata.obsm:
        raise SystemExit("obsm['spatial'] missing")
    if 'lognorm' in adata.layers:
        adata.X = adata.layers['lognorm'].copy()

    adata.obs['global_x'] = np.asarray(adata.obsm['spatial'])[:, 0].astype('float64')
    adata.obs['global_y'] = np.asarray(adata.obsm['spatial'])[:, 1].astype('float64')

    # one (age, sex) per section — the reference is matched on these in step 1
    meta = (adata.obs[[SECTION_COL, 'age', 'sex']].astype(str)
            .groupby(SECTION_COL, observed=True).agg(lambda s: sorted(set(s))))
    multi = meta[(meta['age'].str.len() > 1) | (meta['sex'].str.len() > 1)]
    if len(multi):
        raise SystemExit(f'Sections with more than one age/sex: {list(multi.index)}')

    section_ids = adata.obs[SECTION_COL].astype(str).values
    rows = []
    for i, sid in enumerate(sorted(meta.index), 1):
        out = os.path.join(SECTION_DIR, f'{sid}.h5ad')
        mask = section_ids == sid
        rows.append({'sid': sid, 'age': meta.loc[sid, 'age'][0], 'sex': meta.loc[sid, 'sex'][0],
                     'n_cells': int(mask.sum())})
        if os.path.exists(out) and not args.overwrite:
            print(f'  [{i}/{len(meta)}] {sid}: exists, skipping')
            continue
        sub = adata[mask].copy()
        if not sub.obs_names.is_unique:
            raise ValueError(f'{sid}: obs_names are not unique')
        sub.obs_names = sub.obs_names.astype(str)
        sub.write_h5ad(out, compression='gzip')
        print(f'  [{i}/{len(meta)}] {sid}: {sub.n_obs:,} cells -> {out}')

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(SECTION_DIR, 'sections.tsv'), sep='\t', index=False)
    print(f'\n{len(tab)} sections; table -> {os.path.join(SECTION_DIR, "sections.tsv")}')
    print(tab.to_string(index=False))


if __name__ == '__main__':
    main()
