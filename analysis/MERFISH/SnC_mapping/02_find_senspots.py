#!/usr/bin/env python
# ============================================================================================
# Step 2 — senescent-cell DBSCAN clusters and SenSpots
# --------------------------------------------------------------------------------------------
# Per section, in physical coordinates (µm):
#   1. DBSCAN (eps = 50 µm, min_samples = 5) on the cells called senescent in step 1.
#      Senescent cells assigned to no cluster are 'isolated'.
#   2. Each DBSCAN cluster is grown into a SenSpot: every cell in the section (any cell type,
#      labelled or not) whose nearest clustered senescent cell lies within eps joins that
#      cell's SenSpot. One eps defines both the senescent core and the tissue around it; each
#      cell belongs to at most one SenSpot.
#
# RUN
#   python -u 02_find_senspots.py                        # every section, top-n 300, eps 50
#   python -u 02_find_senspots.py --sid adata126_1250    # one section
#   python -u 02_find_senspots.py --eps 40               # another spatial scale
#
# OUTPUT  OUT_DIR/senspots/topn{n}_eps{eps}/senspots_{sid}.csv.gz
#             index cell id; celltype_final, global_x, global_y, age, sex, sample,
#             SnC_from_sc, SnC_from_sc_score, senescent_dbscan_cluster, in_senspot,
#             senspot_cluster
#         OUT_DIR/senspots/topn{n}_eps{eps}/plots/senspots_{sid}.png
#         OUT_DIR/senspots/topn{n}_eps{eps}/summary/summary_{sid}.csv
#         OUT_DIR/senspots/topn{n}_eps{eps}/senspot_summary.csv   (when run on all sections)
# ============================================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snc_utils as su

# ════════════════════════════════════════════════════════════════════════════════════════════
#  PATHS  —  edit before running (same values as in 01_map_snc.py)
# ════════════════════════════════════════════════════════════════════════════════════════════
SECTION_DIR = '/path/to/output/sections'          # from 00_export_sections.py
OUT_DIR     = '/path/to/output/SnC_mapping'       # step 1 wrote OUT_DIR/labels/ here

KEEP_OBS = ['sample', 'age', 'sex', su.MERFISH_CELLTYPE_COL, 'global_x', 'global_y']


def plot_section(obs, xy, title, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xy[:, 0], xy[:, 1], s=0.2, c='lightgrey', rasterized=True, label='other cells')
    m = obs[su.SENSPOT_MEMBER_COL].values
    ax.scatter(xy[m, 0], xy[m, 1], s=0.6, c='#F39B7F', rasterized=True, label='SenSpot')
    snc = (obs[su.SNC_OUT_COL].astype(str) == su.SNC_CLASS1).values
    core = obs[su.DBSCAN_CLUSTER_COL].notna().values
    ax.scatter(xy[snc & ~core, 0], xy[snc & ~core, 1], s=1.5, c='#4DBBD5', rasterized=True,
               label='isolated SnC')
    ax.scatter(xy[core, 0], xy[core, 1], s=1.5, c='#B2182B', rasterized=True,
               label='clustered SnC')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.legend(markerscale=8, fontsize=7, loc='upper right', frameon=False)
    fig.savefig(path, dpi=200, bbox_inches='tight'); plt.close(fig)


def senspots_section(sid, age, sex, top_n, eps, min_samples, out_dir):
    lab_csv = os.path.join(OUT_DIR, 'labels', f'snc_labels_{sid}.csv.gz')
    if not os.path.exists(lab_csv):
        print(f'  {sid}: no labels (run 01_map_snc.py first), skipping')
        return None
    labels = pd.read_csv(lab_csv, index_col=0)
    labels.index = labels.index.astype(str)
    lab_col, score_col = su.snc_label_cols(top_n)
    if lab_col not in labels.columns:
        print(f'  {sid}: labels have no top_n={top_n} column, skipping')
        return None

    ad = sc.read_h5ad(os.path.join(SECTION_DIR, f'{sid}.h5ad'), backed='r')
    obs = ad.obs[[c for c in KEEP_OBS if c in ad.obs.columns]].copy()
    obs.index = obs.index.astype(str)
    xy = np.asarray(ad.obsm['spatial'], dtype=float)
    ad.file.close()

    obs[su.SNC_OUT_COL] = pd.Categorical(labels[lab_col].reindex(obs.index).values)
    obs[f'{su.SNC_OUT_COL}_score'] = pd.to_numeric(labels[score_col].reindex(obs.index).values,
                                                   errors='coerce')

    obs = su.find_dbscan_clusters(obs, xy, su.SNC_OUT_COL, eps=eps, min_samples=min_samples)
    obs = su.grow_senspots(obs, xy, eps=eps)

    obs.to_csv(os.path.join(out_dir, f'senspots_{sid}.csv.gz'))
    plot_section(obs, xy, f'{sid} ({age}, {sex}) — top-n {top_n}, eps {eps} µm',
                 os.path.join(out_dir, 'plots', f'senspots_{sid}.png'))

    n_snc = int((obs[su.SNC_OUT_COL].astype(str) == su.SNC_CLASS1).sum())
    n_clust = int(obs[su.DBSCAN_CLUSTER_COL].nunique(dropna=True))
    n_core = int(obs[su.DBSCAN_CLUSTER_COL].notna().sum())
    n_spot = int(obs[su.SENSPOT_MEMBER_COL].sum())
    row = {'sid': sid, 'age': age, 'sex': sex, 'top_n': top_n, 'eps': eps,
           'n_cells': len(obs), 'n_senescent': n_snc, 'n_dbscan_clusters': n_clust,
           'n_clustered_senescent': n_core, 'n_isolated_senescent': n_snc - n_core,
           'frac_senescent_clustered': n_core / n_snc if n_snc else np.nan,
           'n_cells_in_senspots': n_spot}
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, 'summary', f'summary_{sid}.csv'), index=False)
    print(f'  {sid}: {n_snc:,} senescent -> {n_clust} DBSCAN clusters holding {n_core:,} '
          f'({n_snc - n_core:,} isolated); {n_spot:,} cells in SenSpots')
    return row


def main():
    p = argparse.ArgumentParser(description='DBSCAN clusters of senescent cells and SenSpots.')
    p.add_argument('--sid', default=None, help='one section id (default: all in sections.tsv)')
    p.add_argument('--top-n', type=int, default=su.TOP_N, help='which step-1 top-n to use')
    p.add_argument('--eps', type=float, default=su.DBSCAN_EPS, help='µm; DBSCAN + growth radius')
    p.add_argument('--min-samples', type=int, default=su.DBSCAN_MIN_SAMPLES)
    args = p.parse_args()

    bad = [n for n, v in [('SECTION_DIR', SECTION_DIR), ('OUT_DIR', OUT_DIR)]
           if v.startswith('/path/to')]
    if bad:
        raise SystemExit(f'ERROR: set the placeholder path(s) at the top of this script: {bad}')

    eps_tag = int(args.eps) if float(args.eps).is_integer() else args.eps
    out_dir = os.path.join(OUT_DIR, 'senspots', f'topn{args.top_n}_eps{eps_tag}')
    for sub in ('', 'plots', 'summary'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    sections = su.read_sections(os.path.join(SECTION_DIR, 'sections.tsv'))
    if args.sid is not None:
        sections = sections[sections['sid'] == args.sid]
        if sections.empty:
            raise SystemExit(f'{args.sid!r} not in sections.tsv')

    print(f'top-n {args.top_n} | eps {args.eps} µm | min_samples {args.min_samples}')
    rows = [senspots_section(r.sid, r.age, r.sex, args.top_n, args.eps, args.min_samples, out_dir)
            for r in sections.itertuples(index=False)]

    if args.sid is None and all(r is not None for r in rows):
        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(out_dir, 'senspot_summary.csv'), index=False)
        t = summ[['n_cells', 'n_senescent', 'n_dbscan_clusters', 'n_clustered_senescent',
                  'n_cells_in_senspots']].sum()
        print('\nAll sections:\n' + t.to_string())
    print('\nDone.')


if __name__ == '__main__':
    main()
