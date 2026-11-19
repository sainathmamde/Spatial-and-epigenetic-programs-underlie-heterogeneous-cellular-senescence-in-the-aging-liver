#!/usr/bin/env python
# ============================================================================================
# Step 1 — senescent-cell (SnC) label transfer from the snRNA-seq reference to MERFISH
# --------------------------------------------------------------------------------------------
# Per section, and within each cell type separately (MERFISH celltype_final == reference
# celltype), against reference nuclei of the SAME age group and sex:
#   1. cosine similarity over the SDTMC signature genes between every reference nucleus and
#      every MERFISH cell; each nucleus is linked to its top-n MERFISH cells (n = 300), and
#      links with similarity < 0.2 are dropped
#   2. P(senescent) per MERFISH cell = summed similarity from senescent nuclei / summed
#      similarity from all linked nuclei
#   3. cells with < 3 links stay unlabelled; among the rest, the top round(prior x N) by
#      P(senescent) are called senescent, where prior = senescent fraction of that cell type
#      in the age- and sex-matched reference (prevalence matching)
#
# RUN
#   python -u 01_map_snc.py                       # every section in sections.tsv, top-n 300
#   python -u 01_map_snc.py --sid adata126_1250   # one section (e.g. one Slurm array task)
#   python -u 01_map_snc.py --top-n 20,40,60,80,100,150,200,250,300,350,400,450,500   # sweep
#
# OUTPUT  OUT_DIR/labels/snc_labels_{sid}.csv.gz
#             index cell id; celltype_final; per top-n:
#             SnC_from_sc_topn{n}        senescent / non-senescent / empty (unlabelled)
#             SnC_from_sc_score_topn{n}  P(senescent); empty if the cell received no links
#         OUT_DIR/labels/details/transfer_{sid}_topn{n}.csv.gz
#             per linked cell: w_1, w_2, total_w, prob1, n_neighbors, eligible, assigned_label,
#             celltype, prior, k_assigned, N_pool
#         OUT_DIR/labels/summary/summary_{sid}.csv
#             per cell type x top-n: reference nuclei, prior, MERFISH cells, called, senescent
# ============================================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snc_utils as su

# ════════════════════════════════════════════════════════════════════════════════════════════
#  PATHS  —  edit before running
# ════════════════════════════════════════════════════════════════════════════════════════════
SECTION_DIR   = '/path/to/output/sections'                 # from 00_export_sections.py
REFERENCE_H5AD = '/path/to/reference_snRNA_counts.h5ad'    # raw counts; obs celltype/age/sex
SNC_META_CSV  = '/path/to/reference_senescence_labels.csv' # index = reference cell id;
                                                           # column predicted_senescence_label
SDTMC_GENES   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SDTMC_147_genes.txt')
OUT_DIR       = '/path/to/output/SnC_mapping'


def check_paths():
    bad = [n for n, v in [('SECTION_DIR', SECTION_DIR), ('REFERENCE_H5AD', REFERENCE_H5AD),
                          ('SNC_META_CSV', SNC_META_CSV), ('OUT_DIR', OUT_DIR)]
           if v.startswith('/path/to')]
    if bad:
        raise SystemExit(f'ERROR: set the placeholder path(s) at the top of this script: {bad}')


def map_section(sid, age, sex, top_n_list, genes, ref_all, args):
    print(f'\n=== {sid} ({age}, {sex}) ===')
    ad_sp_all = sc.read_h5ad(os.path.join(SECTION_DIR, f'{sid}.h5ad'))
    ad_sp_all.obs_names = ad_sp_all.obs_names.astype(str)
    ad_sc_all = ref_all[(ref_all.obs['sex'] == sex) & (ref_all.obs['age'] == age), :]
    print(f'  MERFISH cells {ad_sp_all.n_obs:,} | matched reference nuclei {ad_sc_all.n_obs:,}')

    ref_cts = set(ad_sc_all.obs[su.REF_CELLTYPE_COL].astype(str))
    sp_cts = set(ad_sp_all.obs[su.MERFISH_CELLTYPE_COL].astype(str))
    shared = sorted(ref_cts & sp_cts)
    if sp_cts - ref_cts:
        print(f'  MERFISH-only cell types (left unlabelled): {sorted(sp_cts - ref_cts)}')
    if not shared:
        raise ValueError('No shared cell types between MERFISH and the reference.')

    labels = pd.DataFrame(index=ad_sp_all.obs_names.copy())
    labels[su.MERFISH_CELLTYPE_COL] = ad_sp_all.obs[su.MERFISH_CELLTYPE_COL].astype(str).values
    for n in top_n_list:
        lab_col, score_col = su.snc_label_cols(n)
        labels[lab_col] = pd.Series(dtype=object)
        labels[score_col] = np.nan

    details = {n: [] for n in top_n_list}
    summary = []
    for ct in shared:
        ad_sc = ad_sc_all[ad_sc_all.obs[su.REF_CELLTYPE_COL] == ct, :].copy()
        ad_sp = ad_sp_all[ad_sp_all.obs[su.MERFISH_CELLTYPE_COL] == ct, :].copy()
        if ad_sc.n_obs == 0 or ad_sp.n_obs == 0 or ad_sc.obs[su.REF_SNC_COL].notna().sum() == 0:
            print(f'  {ct}: no reference nuclei / MERFISH cells / senescence labels, skipping')
            continue
        for n in top_n_list:
            pairs = su.compute_topn_pairs(ad_sc, ad_sp, genes, top_n=n, cos_thr=args.cos_thr)
            lab, score, res = su.transfer_prevalence_matched(
                ad_sc, ad_sp, pairs, min_neighbors=args.min_neighbors)
            lab_col, score_col = su.snc_label_cols(n)
            labels.loc[ad_sp.obs_names, lab_col] = lab.values
            labels.loc[ad_sp.obs_names, score_col] = score.values
            if len(res):
                details[n].append(res.assign(celltype=ct))

            called = int(lab.notna().sum())
            n_snc = int((lab == su.SNC_CLASS1).sum())
            prior = float(res['prior'].iloc[0]) if len(res) else np.nan
            summary.append({'sid': sid, 'age': age, 'sex': sex, 'celltype': ct, 'top_n': n,
                            'n_ref': ad_sc.n_obs, 'prior_senescent': prior,
                            'n_merfish': ad_sp.n_obs, 'n_called': called, 'n_senescent': n_snc})
            print(f'  {ct:<24} top_n={n:<4} called {called:,}/{ad_sp.n_obs:,}, senescent {n_snc:,}'
                  + (f' ({n_snc / called:.1%} of called)' if called else ''))

    out_lab = os.path.join(OUT_DIR, 'labels', f'snc_labels_{sid}.csv.gz')
    labels.to_csv(out_lab)
    for n, parts in details.items():
        if parts:
            pd.concat(parts).to_csv(os.path.join(OUT_DIR, 'labels', 'details',
                                                 f'transfer_{sid}_topn{n}.csv.gz'))
    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'labels', 'summary',
                                              f'summary_{sid}.csv'), index=False)
    print(f'  -> {out_lab}')


def main():
    p = argparse.ArgumentParser(description='SnC label transfer, per section and cell type.')
    p.add_argument('--sid', default=None, help='one section id (default: all in sections.tsv)')
    p.add_argument('--top-n', default=str(su.TOP_N), help='comma-separated top-n value(s)')
    p.add_argument('--cos-thr', type=float, default=su.COS_THR)
    p.add_argument('--min-neighbors', type=int, default=su.MIN_NEIGHBORS)
    args = p.parse_args()
    check_paths()

    top_n_list = [int(x) for x in args.top_n.split(',')]
    for sub in ('labels', 'labels/details', 'labels/summary'):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    sections = su.read_sections(os.path.join(SECTION_DIR, 'sections.tsv'))
    if args.sid is not None:
        sections = sections[sections['sid'] == args.sid]
        if sections.empty:
            raise SystemExit(f'{args.sid!r} not in sections.tsv')

    genes = su.read_gene_list(SDTMC_GENES)
    print(f'SDTMC genes: {len(genes)} | top-n: {top_n_list} | cos_thr: {args.cos_thr} | '
          f'min_neighbors: {args.min_neighbors}')
    ref_all = su.load_snc_reference(REFERENCE_H5AD, SNC_META_CSV)

    for r in sections.itertuples(index=False):
        map_section(r.sid, r.age, r.sex, top_n_list, genes, ref_all, args)

    if args.sid is None:
        summ = pd.concat([pd.read_csv(os.path.join(OUT_DIR, 'labels', 'summary', f'summary_{s}.csv'))
                          for s in sections['sid']])
        summ.to_csv(os.path.join(OUT_DIR, 'snc_mapping_summary.csv'), index=False)
        tot = summ.groupby('top_n')[['n_merfish', 'n_called', 'n_senescent']].sum()
        print('\nAll sections:\n' + tot.to_string())
    print('\nDone.')


if __name__ == '__main__':
    main()
