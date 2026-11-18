#!/usr/bin/env python
# ============================================================================================
# Step 2 — MERFISH two-stage cell-type annotation by CCA co-embedding with a snRNA reference
# --------------------------------------------------------------------------------------------
# Annotates every wild-type MERFISH cell that passes QC in 01_merfish_qc.py against a
# single-nucleus reference using the recipe of the whole-mouse-brain MERFISH atlas
# (Zhang et al., Nature 2023): Seurat-v3 CCA + mutual-nearest-neighbour anchors, run as two
# hierarchical rounds
#   STAGE 1  Hepatocytes vs Non-hepatocytes         (all cells, all shared genes)
#   STAGE 2  the 6 non-hepatocyte cell types        (non-hep cells, HVG re-selected)
# then combines the two with an adjusted confidence (s1_conf x s2_conf), marks cells below
# the 0.8 cutoff in either stage as 'Unknown', and re-embeds the confident cells alone.
# The CCA/MNN maths comes from the public ALLCools package
# (`ALLCools.integration.SeuratIntegration`).
#
# --------------------------------------------------------------------------------------------
# REQUIREMENTS
#   • Python env with: scanpy, anndata, numpy, pandas, scikit-learn, scipy, matplotlib.
#     GPU is OPTIONAL — if `rapids_singlecell` (+ cupy/rmm) is importable it is used
#     automatically; otherwise everything falls back to CPU scanpy (slower but identical logic).
#   • The public ALLCools package, for the CCA + MNN-anchor integration:
#         pip install allcools      (source: https://github.com/lhqing/ALLCools , MIT)
#     The pipeline uses only `ALLCools.integration.SeuratIntegration` — the pure-Python
#     Seurat-v3 CCA/MNN reimplementation the Zhuang lab used for their whole-mouse-brain atlas.
#
# INPUTS  (edit the CONFIG block below)
#   • MERFISH_PATH  : merfish_qc_filtered.h5ad from 01_merfish_qc.py — `.layers['counts']` holds
#                     RAW integer counts; obs has 'experiment','sample','age','sex'.
#                     Genes must overlap the reference panel.
#   • REFERENCE_PATH: an .h5ad reference whose `.X` holds RAW counts and obs['celltype'] the
#                     ground-truth labels to transfer.
#
# OUTPUTS  ->  OUT_DIR/
#   stage1/ , stage2/         per-stage plots (co-embedding UMAPs, MERFISH UMAPs, marker
#                             dotplots, confidence histogram, reference-correlation heatmaps)
#   final/                    combined labels, the FINAL MERFISH-alone UMAP, composition
#   merfish_annotated.h5ad    all cells + per-stage labels/confidences + embeddings
#   merfish_final_known.h5ad  confidently-annotated cells only (the deliverable object)
#
# RUN
#   python -u 02_twostage_cca_annotation.py
#   (needs ~350 GB RAM at the default CCA_MAX_CC_CELLS — see the MEMORY note above
#    cca_coembed_transfer; a GPU is optional but strongly recommended at millions of cells)
# ============================================================================================

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import spearmanr

# ── GPU (optional): rapids_singlecell accelerates PCA / neighbours / Leiden / UMAP ──────────
try:
    import rapids_singlecell as rsc
    import cupy as cp                      # noqa: F401
    import rmm
    rmm.reinitialize(managed_memory=True, pool_allocator=False)
    USE_GPU = True
    print('rapids_singlecell available — GPU mode ON')
except Exception as e:
    print(f'rapids_singlecell unavailable ({type(e).__name__}); using scanpy CPU fallback')
    USE_GPU = False


# ════════════════════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit for your dataset
# ════════════════════════════════════════════════════════════════════════════════════════════
MERFISH_PATH   = '/path/to/output/QC/merfish_qc_filtered.h5ad'   # output of 01_merfish_qc.py
REFERENCE_PATH = '/path/to/reference_snRNA_counts.h5ad'          # raw counts + obs['celltype']
OUT_DIR        = '/path/to/output/twostage_cca_annotation'

# Confidence cutoffs (a cell below its stage's cutoff becomes 'Unknown'). 0.8 == the paper's
# subclass-level threshold. One flat value per stage; both stages use 0.8.
S1_CUTOFF = 0.80
S2_CUTOFF = 0.80

# Batch key used only for the FINAL MERFISH-alone Harmony re-embedding (harmony is skipped
# automatically if the key has <2 levels). The CCA transfer itself does NOT use it.
BATCH_KEY = 'experiment'

# Pipeline dimensions (kept identical across CCA components, label transfer, Leiden and UMAP).
TARGET_SUM   = 1000      # per-modality normalize_total before CCA (Zhuang used 1000)
N_PCS        = 50        # CCA components == transfer PCs == Leiden/UMAP PCs
MIN_DIST     = 0.5       # UMAP min_dist
ANNOT_RES    = 2.0       # Leiden resolution
HARMONY_THETA = 2.0
N_NEIGH      = 30        # kNN for neighbours graphs
UMAP_INIT    = 'spectral'   # cuML supports 'spectral'/'random'; spectral preserves global layout
CCA_MAX_CC_CELLS = 100_000  # CCA anchor subsample (drives peak RAM — see cca_coembed_transfer)

# Marker genes for the dotplots (only those present in the panel are used).
MARKERS = {
    'Hepatocyte': ['Hnf4a'], 'Hep(zonation)': ['Cyp2f2', 'Gstm3', 'Mt2', 'Aspg'],
    'Cholangiocytes': ['Epcam'], 'Hepatic_stellate_cells': ['Dcn', 'Tcf21'],
    'Endothelial': ['Lyve1', 'Aqp1', 'Ptprb'], 'BCells': ['Cd79a'],
    'Myeloid': ['Clec4f', 'Vsig4', 'Cd5l', 'Spi1', 'Lyz2'], 'TCells': ['Nkg7'],
}


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Small helpers
# ════════════════════════════════════════════════════════════════════════════════════════════
def _dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def _p(subdir, fname):
    d = os.path.join(OUT_DIR, subdir) if subdir else OUT_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def group_mean(X, genes, labels, exclude='Unknown'):
    """genes x types DataFrame of mean expression, one column per (non-excluded) label."""
    labels = np.asarray(labels).astype(str)
    types = sorted(t for t in set(labels) if t != exclude)
    return pd.DataFrame({t: np.asarray(_dense(X)[labels == t].mean(0)).ravel() for t in types},
                        index=list(genes))


def markers_present(panel_genes):
    s = set(panel_genes)
    return {ct: [g for g in gs if g in s] for ct, gs in MARKERS.items()
            if any(g in s for g in gs)}


def known(ad, col):
    """View of the cells whose label in `col` is not 'Unknown'."""
    return ad[ad.obs[col].astype(str) != 'Unknown']


def normalize_merfish(ad):
    """counts -> total-count normalize (1e4) -> log1p, kept in X and layers['lognorm']."""
    ad.X = ad.layers['counts'].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    ad.layers['lognorm'] = ad.X.copy()


def save_object(ad, fname):
    for c in ad.obs.columns:
        if ad.obs[c].dtype == object or hasattr(ad.obs[c], 'cat'):
            ad.obs[c] = ad.obs[c].astype(str)
    ad.write_h5ad(os.path.join(OUT_DIR, fname))
    print(f'  Saved -> {fname}')


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Plot primitives
# ════════════════════════════════════════════════════════════════════════════════════════════
def save_umap(ad, cols, fname, subdir='', title=''):
    """Multi-panel UMAP, one panel per obs column (categorical: coloured; numeric: colourbar)."""
    cols = [c for c in cols if c in ad.obs.columns]
    if not cols or 'X_umap' not in ad.obsm:
        return
    nc = min(3, len(cols)); nr = int(np.ceil(len(cols) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr)); axes = np.array(axes).flatten()
    xy = ad.obsm['X_umap']; pal = plt.cm.tab20.colors
    for ax, c in zip(axes, cols):
        v = ad.obs[c]
        if pd.api.types.is_numeric_dtype(v):
            s = ax.scatter(xy[:, 0], xy[:, 1], c=v.values, s=0.5, alpha=0.5, rasterized=True)
            plt.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
        else:
            v = v.astype('category')
            for i, cat in enumerate(v.cat.categories):
                m = (v == cat).values
                ax.scatter(xy[m, 0], xy[m, 1], s=0.5, alpha=0.4, rasterized=True,
                           color=pal[i % len(pal)], label=str(cat))
            ax.legend(markerscale=5, fontsize=6, ncol=max(1, v.cat.categories.size // 12))
        ax.set_title(c, fontsize=9)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(_p(subdir, fname), dpi=150, bbox_inches='tight'); plt.close()


def save_dotplot(ad, markers, groupby, fname, subdir='', title=''):
    """Marker dotplot (mean expression + fraction expressing) from log1p(counts)."""
    genes = [g for gs in markers.values() for g in gs]
    if not genes or ad.obs[groupby].nunique() < 2:
        return
    Xc = _dense(ad[:, genes].layers['counts'])
    sub = sc.AnnData(X=np.log1p(np.asarray(Xc, np.float32)), obs=ad.obs[[groupby]].copy(),
                     var=ad[:, genes].var.copy())
    sub.obs[groupby] = sub.obs[groupby].astype('category')
    fs = (max(12, len(genes) * 0.45 + 3), max(4, sub.obs[groupby].nunique() * 0.4 + 2))
    sc.pl.dotplot(sub, var_names=markers, groupby=groupby, use_raw=False, standard_scale='var',
                  cmap='Oranges', dot_max=0.7, dendrogram=False, figsize=fs, show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    fig.savefig(_p(subdir, fname), dpi=150, bbox_inches='tight'); plt.close(fig)


def save_heatmap(df, fname, subdir='', title='', cmap='RdBu_r', vmin=-1, vmax=1, xlabel='', ylabel=''):
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * df.shape[1] + 3), max(4, 0.4 * df.shape[0] + 2)))
    im = ax.imshow(df.values.astype(float), aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=7)
    if df.shape[0] * df.shape[1] <= 600:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(j, i, format(float(df.values[i, j]), '.2f'), ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    plt.tight_layout(); plt.savefig(_p(subdir, fname), dpi=150, bbox_inches='tight'); plt.close()


def confidence_hist(ad, col, fname, subdir='', title=''):
    v = np.asarray(ad.obs[col], dtype=float); v = v[~np.isnan(v)]
    if v.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(v, bins=40, color='steelblue', edgecolor='none')
    ax.axvline(np.median(v), color='red', ls='--', lw=1, label=f'median={np.median(v):.3f}')
    for c in (0.5, 0.8):
        ax.axvline(c, color='grey', ls=':', lw=1); ax.text(c, ax.get_ylim()[1] * 0.95, f' {c}', fontsize=7, color='grey')
    ax.set_xlabel(col); ax.set_ylabel('# cells'); ax.set_title(title or col, fontsize=10); ax.legend()
    plt.tight_layout(); plt.savefig(_p(subdir, fname), dpi=150, bbox_inches='tight'); plt.close()


def proportion_stackbar(obs, group_key, label_col, fname, subdir='', order=None):
    tab = obs.groupby([group_key, label_col], observed=True).size().unstack(fill_value=0)
    if order:
        tab = tab.reindex([o for o in order if o in tab.index])
    pct = tab.div(tab.sum(1), axis=0) * 100
    pct.to_csv(_p(subdir, fname.replace('.pdf', '.csv')))
    allt = sorted(pct.columns); pal = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    colors = {c: pal[i % len(pal)] for i, c in enumerate(allt)}
    fig, ax = plt.subplots(figsize=(max(8, len(pct) * 0.9 + 3), 6)); bottoms = np.zeros(len(pct))
    for c in allt:
        ax.bar(range(len(pct)), pct[c].values, bottom=bottoms, color=colors[c], label=c, width=0.8)
        bottoms += pct[c].values
    ax.set_xticks(range(len(pct))); ax.set_xticklabels(pct.index.astype(str), rotation=45, ha='right')
    ax.set_ylabel('Proportion (%)'); ax.set_title(f'{label_col} by {group_key}')
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(_p(subdir, fname), dpi=150, bbox_inches='tight'); plt.close()


def correlation_heatmap(mer, label_col, genes, ref_centroids, fname, subdir='', title=''):
    """Pearson + Spearman correlation of per-type MERFISH centroids vs the reference centroids,
       over `genes`. A good diagonal (each MERFISH type most-correlated with its own reference
       type) is the sanity check that the transfer is coherent."""
    genes = [g for g in genes if g in ref_centroids.index]
    if not genes:
        return
    mcen = group_mean(mer[:, genes].layers['lognorm'], genes, mer.obs[label_col].values)
    rc = ref_centroids.loc[genes]
    for method in ('pearson', 'spearman'):
        corr = pd.DataFrame(index=mcen.columns, columns=rc.columns, dtype=float)
        for mt in mcen.columns:
            for rt in rc.columns:
                corr.loc[mt, rt] = (np.corrcoef(mcen[mt].values, rc[rt].values)[0, 1] if method == 'pearson'
                                    else spearmanr(mcen[mt].values, rc[rt].values).correlation)
        corr = corr.astype(float)
        fn = fname.replace('.pdf', f'_{method}.pdf')
        save_heatmap(corr, fn, subdir, f'{title} [{method}]', xlabel='reference type', ylabel='MERFISH type')
        corr.to_csv(_p(subdir, fn.replace('.pdf', '.csv')))


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Embedding / clustering (GPU if available, else scanpy)
# ════════════════════════════════════════════════════════════════════════════════════════════
def _umap_call(a, min_dist, init_pos=UMAP_INIT):
    if not USE_GPU:
        sc.tl.umap(a, min_dist=min_dist); return
    try:
        rsc.tl.umap(a, min_dist=min_dist, init_pos=init_pos)
    except Exception as e:                      # spectral init can fail at scale -> random
        print(f"    UMAP init_pos='{init_pos}' failed ({type(e).__name__}); using 'random'")
        rsc.tl.umap(a, min_dist=min_dist, init_pos='random')


def _usable_batch_keys(e, batch_key):
    keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
    return [k for k in keys if k in e.obs and e.obs[k].nunique() > 1]


def embed(ad, genes, batch_key, min_dist=MIN_DIST, n_pcs_use=N_PCS, do_umap=True):
    """scale -> PCA -> Harmony(batch_key) -> neighbours [-> UMAP]. Writes obsm rep (and X_umap).
       Harmony is skipped when batch_key has <2 levels (rep is then plain 'X_pca')."""
    e = ad[:, genes].copy(); e.X = e.layers['lognorm'].copy()
    sc.pp.scale(e, max_value=10)
    keys = _usable_batch_keys(e, batch_key)
    hkey = keys if len(keys) > 1 else (keys[0] if keys else None)
    if keys:
        print(f"  Harmony covariates: {keys} (theta={HARMONY_THETA})")
    if USE_GPU:
        rsc.get.anndata_to_GPU(e); rsc.pp.pca(e, n_comps=N_PCS)
        if hkey is not None:
            try:
                rsc.pp.harmony_integrate(e, key=hkey, basis='X_pca', adjusted_basis='X_pca_harmony',
                                         max_iter_harmony=50, theta=HARMONY_THETA)
            except TypeError:
                rsc.pp.harmony_integrate(e, key=hkey, basis='X_pca', adjusted_basis='X_pca_harmony',
                                         max_iter_harmony=50)
            rep = 'X_pca_harmony'
        else:
            rep = 'X_pca'
        rsc.pp.neighbors(e, n_neighbors=N_NEIGH, use_rep=rep, n_pcs=n_pcs_use)
        if do_umap:
            _umap_call(e, min_dist)
        rsc.get.anndata_to_CPU(e)
    else:
        sc.tl.pca(e, n_comps=N_PCS, svd_solver='randomized')
        if hkey is not None:
            sc.external.pp.harmony_integrate(e, key=hkey, basis='X_pca', adjusted_basis='X_pca_harmony',
                                             max_iter_harmony=50, theta=HARMONY_THETA)
            rep = 'X_pca_harmony'
        else:
            rep = 'X_pca'
        sc.pp.neighbors(e, n_neighbors=N_NEIGH, use_rep=rep, n_pcs=n_pcs_use)
        if do_umap:
            sc.tl.umap(e, min_dist=min_dist)
    if do_umap:
        ad.obsm['X_umap'] = e.obsm['X_umap']
    ad.obsm[rep] = np.asarray(e.obsm[rep], dtype=np.float32)
    del e; gc.collect()
    return rep


def cluster_umap_on_rep(coords, res=ANNOT_RES, n_pcs=None, min_dist=MIN_DIST):
    """ONE neighbour graph -> BOTH Leiden and UMAP (so clustering and layout share dimensions)."""
    a = sc.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    a.obsm['X_rep'] = np.asarray(coords, dtype=np.float32)
    nd = coords.shape[1] if n_pcs is None else min(n_pcs, coords.shape[1])
    print(f"    cluster+umap on {coords.shape[0]:,} cells x {nd} dims (res={res}, min_dist={min_dist})")
    if USE_GPU:
        rsc.get.anndata_to_GPU(a)
        rsc.pp.neighbors(a, n_neighbors=N_NEIGH, use_rep='X_rep', n_pcs=n_pcs)
        rsc.tl.leiden(a, resolution=res, key_added='leiden')
        _umap_call(a, min_dist)
        rsc.get.anndata_to_CPU(a)
    else:
        sc.pp.neighbors(a, n_neighbors=N_NEIGH, use_rep='X_rep', n_pcs=n_pcs)
        sc.tl.leiden(a, resolution=res, key_added='leiden')
        sc.tl.umap(a, min_dist=min_dist)
    return a.obs['leiden'].astype(str).values, np.asarray(a.obsm['X_umap'])


def umap_on_rep(ad, rep, min_dist=MIN_DIST, n_pcs=N_PCS):
    if USE_GPU:
        rsc.get.anndata_to_GPU(ad)
        rsc.pp.neighbors(ad, n_neighbors=N_NEIGH, use_rep=rep, n_pcs=n_pcs)
        _umap_call(ad, min_dist)
        rsc.get.anndata_to_CPU(ad)
    else:
        sc.pp.neighbors(ad, n_neighbors=N_NEIGH, use_rep=rep, n_pcs=n_pcs)
        sc.tl.umap(ad, min_dist=min_dist)


# ════════════════════════════════════════════════════════════════════════════════════════════
#  CCA + MNN-anchor co-embedding and label transfer  (the ALLCools / Seurat-v3 core)
#
#  ⚠ MEMORY: peak is set by CCA_MAX_CC_CELLS, NOT by the number of cells. ALLCools downsamples
#  each modality to max_cc_cells and forms a DENSE (max_cc_cells x max_cc_cells) float64 matrix:
#      100_000 -> ~80 GB   |   50_000 -> ~20 GB   |   20_000 -> ~3 GB
#  All cells are then projected through it in chunks (lowering it approximates, never drops
#  cells). 100_000 matches the paper; request ~300-350 GB RAM for millions of cells at that
#  setting, or lower CCA_MAX_CC_CELLS.
# ════════════════════════════════════════════════════════════════════════════════════════════
def cca_coembed_transfer(mer, ref_counts, ref_labels, genes, hvg_select=False, tag=''):
    """Co-embed a MERFISH subset with a reference subset and transfer `ref_labels`.
       Per-modality normalize->log->scale BEFORE merging (Zhuang: better co-embedding).
       Returns (pred, conf, merged); merged.obsm['X_pca_integrate'], merged.obs['modality'] set.
       hvg_select=True reproduces the paper's round-2 per-partition HVG re-selection."""
    # Seurat-v3 CCA + MNN anchors, from the public ALLCools package (Hanqing Liu / Ecker lab).
    # This is the exact import Zhang et al. 2023 use in their published integration_round1.ipynb:
    #   pip install allcools     (or:  pip install git+https://github.com/lhqing/ALLCools)
    from ALLCools.integration.seurat_class import SeuratIntegration
    genes = list(genes)
    a_seq = sc.AnnData(X=ref_counts[:, genes].X.copy(),
                       obs=pd.DataFrame({'celltype': np.asarray(ref_labels).astype(str)},
                                        index=[f'REF_{i}' for i in range(ref_counts.n_obs)]),
                       var=pd.DataFrame(index=genes))
    a_mer = sc.AnnData(X=mer[:, genes].layers['counts'].copy(),
                       obs=pd.DataFrame(index=list(mer.obs_names)),
                       var=pd.DataFrame(index=genes))
    for a in (a_seq, a_mer):
        sc.pp.normalize_total(a, target_sum=TARGET_SUM); sc.pp.log1p(a)
    if hvg_select:
        sc.pp.highly_variable_genes(a_seq)
        hv = [g for g, d in zip(a_seq.var_names, a_seq.var['dispersions']) if d > 0]
        if len(hv) >= 20:
            a_seq = a_seq[:, hv].copy(); a_mer = a_mer[:, hv].copy()
            print(f"    {tag}HVG re-selection (round-2): {len(hv)}/{len(genes)} genes")
    for a in (a_seq, a_mer):
        sc.pp.scale(a)
    merged = a_seq.concatenate(a_mer, batch_categories=['seq', 'merfish'],
                               batch_key='modality', index_unique=None)
    print(f"    {tag}merge: seq {a_seq.n_obs:,} + merfish {a_mer.n_obs:,} = {merged.n_obs:,}")
    npc = int(min(N_PCS, merged.shape[1] - 1))
    del a_seq, a_mer; gc.collect()
    sc.tl.pca(merged, svd_solver='arpack', n_comps=npc)
    alist = [merged[merged.obs['modality'] == 'seq'], merged[merged.obs['modality'] == 'merfish']]
    integ = SeuratIntegration()
    print(f"    {tag}find_anchor (CCA + MNN) ...")
    integ.find_anchor(alist, k_local=None, key_local='X_pca', k_anchor=5, key_anchor='X',
                      dim_red='cca', max_cc_cells=CCA_MAX_CC_CELLS, k_score=30, k_filter=None,
                      scale1=False, scale2=False, n_components=npc, n_features=200,
                      alignments=[[[0], [1]]])
    print(f"    {tag}label_transfer ...")
    tr = integ.label_transfer(ref=[0], qry=[1], categorical_key=['celltype'], key_dist='X_pca',
                              k_weight=100, npc=npc)
    pred = tr['celltype'].idxmax(axis=1).values.astype(object)
    conf = tr['celltype'].max(axis=1).values.astype(np.float32)
    print(f"    {tag}integrate (corrected joint space) ...")
    corrected = integ.integrate(key_correct='X_pca', row_normalize=True, n_components=npc,
                                k_weight=100, sd=1, alignments=[[[0], [1]]])
    merged.obsm['X_pca_integrate'] = np.concatenate(corrected).astype(np.float32)
    return pred, conf, merged


def apply_cutoff(pred, conf, cutoff):
    """Cells below `cutoff` confidence become 'Unknown'."""
    return np.where(np.asarray(conf) >= float(cutoff), np.asarray(pred).astype(object), 'Unknown').astype(object)


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Co-embedding label bookkeeping + per-stage plots
# ════════════════════════════════════════════════════════════════════════════════════════════
def merfish_rep_from_merged(ad, merged, key='X_pca_integrate'):
    """Copy the MERFISH rows of the corrected joint space onto the MERFISH object."""
    m = (merged.obs['modality'] == 'merfish').values
    ad.obsm[key] = np.asarray(merged.obsm[key][m], dtype=np.float32)
    return key


def joint_labels(merged, ad, label_col, ref_true_labels):
    """celltype_joint = TRUE reference label for seq cells, predicted label for MERFISH cells."""
    m = (merged.obs['modality'] == 'merfish').values
    cj = np.empty(merged.n_obs, dtype=object)
    cj[~m] = np.asarray(ref_true_labels).astype(str)
    cj[m] = ad.obs[label_col].astype(str).values
    merged.obs['celltype_joint'] = pd.Categorical(cj.astype(str))
    return merged


def coembed_umaps(merged, subdir, tag):
    """The co-embedding views on the shared X_pca_integrate layout."""
    save_umap(merged, ['modality', 'celltype_joint'], 'coembed_umap_all.pdf', subdir,
              title=f'{tag} — co-embedding (all cells)')
    save_umap(known(merged, 'celltype_joint'), ['modality', 'celltype_joint'],
              'coembed_umap_known.pdf', subdir, title=f'{tag} — co-embedding (Unknown removed)')
    save_umap(merged[merged.obs['modality'] == 'seq'], ['celltype_joint'],
              'coembed_umap_reference_only.pdf', subdir, title=f'{tag} — reference cells')
    mer_only = merged[merged.obs['modality'] == 'merfish']
    save_umap(known(mer_only, 'celltype_joint'), ['celltype_joint'],
              'coembed_umap_merfish_only_known.pdf', subdir, title=f'{tag} — MERFISH (Unknown removed)')


def stage_plots(mer, merged, subdir, label_col, argmax_col, conf_col, cluster_col,
                ref_centroids, marker_genes, mk):
    """Every plot for one stage: co-embedding UMAPs, MERFISH-only UMAPs, marker dotplots,
       confidence histogram, and reference-correlation heatmaps."""
    umap_on_rep(merged, 'X_pca_integrate', min_dist=MIN_DIST, n_pcs=N_PCS)
    coembed_umaps(merged, subdir, subdir)

    cols = [label_col, argmax_col, cluster_col, conf_col, 'age', 'sex']
    save_umap(mer, cols, 'merfish_umap_all.pdf', subdir, title=f'{subdir} — MERFISH (incl. Unknown)')
    save_umap(known(mer, label_col), [c for c in cols if c != conf_col], 'merfish_umap_known.pdf',
              subdir, title=f'{subdir} — MERFISH (Unknown removed)')

    save_dotplot(mer, mk, label_col, 'dotplot_by-celltype_all.pdf', subdir, title=f'{subdir} markers (all)')
    save_dotplot(known(mer, label_col), mk, label_col, 'dotplot_by-celltype_known.pdf', subdir,
                 title=f'{subdir} markers (Unknown removed)')
    save_dotplot(mer, mk, cluster_col, 'dotplot_by-cluster.pdf', subdir, title=f'{subdir} markers by cluster')

    confidence_hist(mer, conf_col, 'confidence_distribution.pdf', subdir, title=f'{subdir} confidence')
    correlation_heatmap(known(mer, label_col), label_col, marker_genes, ref_centroids,
                        'corr_markers.pdf', subdir, title=f'{subdir} markers:')


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════════════════════════════════════
def load_merfish():
    """Read and normalise all cells from the QC-filtered wild-type MERFISH input."""
    ad = sc.read_h5ad(MERFISH_PATH)
    if 'age' in ad.obs:
        ad.obs['age'] = ad.obs['age'].astype(str)
    if 'counts' not in ad.layers:
        raise SystemExit("ERROR: MERFISH object needs layers['counts'] with RAW counts.")
    print(f"Loaded {ad.n_obs:,} QC-filtered cells")
    for c in ['experiment', 'age']:
        if c in ad.obs:
            print(f"  {c}: {dict(ad.obs[c].astype(str).value_counts())}")
    normalize_merfish(ad)
    return ad


def load_reference(genes):
    """Return (ref_counts, REF_CT, ref_centroids): a raw-count reference AnnData over `genes`,
       its per-cell celltype array, and its per-type log-normalised centroids (for correlation)."""
    rc = sc.read_h5ad(REFERENCE_PATH)
    REF_CT = rc.obs['celltype'].astype(str).values
    ref_counts = sc.AnnData(X=rc[:, genes].X.copy(), var=pd.DataFrame(index=genes),
                            obs=pd.DataFrame(index=[f'R{i}' for i in range(rc.n_obs)]))
    # log-normalised centroids for the correlation heatmaps
    rl = rc[:, genes].copy(); sc.pp.normalize_total(rl, target_sum=1e4); sc.pp.log1p(rl)
    ref_centroids = group_mean(rl.X, genes, REF_CT, exclude='___none___')
    del rc, rl; gc.collect()
    print(f"Reference: {ref_counts.n_obs:,} cells; types {sorted(set(REF_CT))}")
    return ref_counts, REF_CT, ref_centroids


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ════════════════════════════════════════════════════════════════════════════════════════════
def check_paths():
    bad = [n for n, p in [('MERFISH_PATH', MERFISH_PATH), ('REFERENCE_PATH', REFERENCE_PATH),
                          ('OUT_DIR', OUT_DIR)] if p.startswith('/path/to')]
    if bad:
        raise SystemExit(f"ERROR: set the placeholder path(s) in the CONFIG block: {bad}")


def main():
    check_paths()
    os.makedirs(OUT_DIR, exist_ok=True)
    print('=' * 70); print('MERFISH two-stage CCA annotation'); print('=' * 70)

    adata = load_merfish()
    ref_vars = sc.read_h5ad(REFERENCE_PATH, backed='r').var_names
    GENES = list(adata.var_names.intersection(ref_vars))
    print(f"Shared genes (panel ∩ reference): {len(GENES)}")
    ref_counts, REF_CT, ref_centroids = load_reference(GENES)
    mk = markers_present(adata.var_names)
    MARKER_GENES = [g for g in dict.fromkeys(x for v in mk.values() for x in v) if g in set(GENES)]

    # ── STAGE 1 : Hepatocytes vs Non-hepatocytes ─────────────────────────────────────────────
    print('\n' + '=' * 70); print('STAGE 1 — Hepatocytes vs NonHep'); print('=' * 70)
    s1_ref = np.where(REF_CT == 'Hepatocytes', 'Hepatocytes', 'NonHep')
    s1_pred, s1_conf, merged1 = cca_coembed_transfer(adata, ref_counts, s1_ref, GENES, tag='[s1] ')
    adata.obs['s1_argmax'] = s1_pred                         # BEFORE cutoff (used to route stage 2)
    adata.obs['s1_conf'] = s1_conf
    adata.obs['s1_celltype'] = apply_cutoff(s1_pred, s1_conf, S1_CUTOFF)
    adata.obs['confidence'] = s1_conf
    print(f"  argmax: {dict(pd.Series(s1_pred).value_counts())}")
    print(f"  after cutoff {S1_CUTOFF}: {dict(pd.Series(adata.obs['s1_celltype']).value_counts())}")

    rep1 = merfish_rep_from_merged(adata, merged1)
    s1_leiden, s1_umap = cluster_umap_on_rep(adata.obsm[rep1], ANNOT_RES, n_pcs=N_PCS, min_dist=MIN_DIST)
    adata.obs['s1_leiden'] = s1_leiden
    adata.obsm['X_umap'] = s1_umap
    merged1 = joint_labels(merged1, adata, 's1_celltype', s1_ref)
    stage_plots(adata, merged1, 'stage1', 's1_celltype', 's1_argmax', 's1_conf', 's1_leiden',
                ref_centroids, MARKER_GENES, mk)
    umap1 = adata.obsm['X_umap'].copy()                     # keep stage-1 layout for final plots
    del merged1; gc.collect()

    # ── STAGE 2 : the 6 non-hepatocyte types ─────────────────────────────────────────────────
    print('\n' + '=' * 70); print('STAGE 2 — non-hepatocyte cell types'); print('=' * 70)
    nh_mask = adata.obs['s1_argmax'].values == 'NonHep'     # route by ARGMAX (no pre-filtering)
    nh = adata[nh_mask].copy()
    ref_nh_mask = REF_CT != 'Hepatocytes'
    ref_nh = ref_counts[ref_nh_mask].copy()
    print(f"  MERFISH non-hep (argmax): {nh.n_obs:,} | reference non-hep: {ref_nh.n_obs:,}")
    s2_pred, s2_conf, merged2 = cca_coembed_transfer(nh, ref_nh, REF_CT[ref_nh_mask], GENES,
                                                     hvg_select=True, tag='[s2] ')
    nh.obs['s2_argmax'] = s2_pred
    nh.obs['s2_conf'] = s2_conf
    nh.obs['s2_celltype'] = apply_cutoff(s2_pred, s2_conf, S2_CUTOFF)
    nh.obs['confidence'] = s2_conf
    print(f"  argmax: {dict(pd.Series(s2_pred).value_counts())}")
    print(f"  after cutoff {S2_CUTOFF}: {dict(pd.Series(nh.obs['s2_celltype']).value_counts())}")

    rep2 = merfish_rep_from_merged(nh, merged2)
    s2_leiden, s2_umap = cluster_umap_on_rep(nh.obsm[rep2], ANNOT_RES, n_pcs=N_PCS, min_dist=MIN_DIST)
    nh.obs['s2_leiden'] = s2_leiden
    nh.obsm['X_umap'] = s2_umap
    merged2 = joint_labels(merged2, nh, 's2_celltype', REF_CT[ref_nh_mask])
    stage_plots(nh, merged2, 'stage2', 's2_celltype', 's2_argmax', 's2_conf', 's2_leiden',
                ref_centroids, MARKER_GENES, mk)
    del merged2; gc.collect()

    # ── FINAL : combine stages + adjusted confidence ─────────────────────────────────────────
    print('\n' + '=' * 70); print('FINAL — combine stages'); print('=' * 70)
    final = np.array(['Unknown'] * adata.n_obs, dtype=object)
    adj = np.asarray(adata.obs['s1_conf'].values, np.float32).copy()
    hep = adata.obs['s1_celltype'].values == 'Hepatocytes'
    final[hep] = 'Hepatocytes'                              # hepatocytes: stage-1 only
    pos = {b: i for i, b in enumerate(adata.obs_names)}
    idx = np.array([pos[b] for b in nh.obs_names])
    s1lab_nh = adata.obs['s1_celltype'].astype(str).values[idx]
    s2lab = nh.obs['s2_celltype'].astype(str).values
    final[idx] = np.where((s1lab_nh == 'Unknown') | (s2lab == 'Unknown'), 'Unknown', s2lab)
    adj[idx] = np.asarray(adata.obs['s1_conf'].values, np.float32)[idx] * np.asarray(nh.obs['s2_conf'].values, np.float32)
    adata.obs['celltype_final'] = final
    adata.obs['adjusted_confidence'] = adj
    print(f"  final: {dict(pd.Series(final).value_counts())}")

    # combined labels on the stage-1 layout + co-embedding views with TRUE 7-type reference labels
    adata.obsm['X_umap'] = umap1
    fcols = ['celltype_final', 's1_celltype', 'adjusted_confidence', 's1_leiden', 'age', 'sex']
    save_umap(adata, fcols, 'final_umap_stage1space_all.pdf', 'final',
              title='Final cell types on the stage-1 layout (all cells)')
    save_umap(known(adata, 'celltype_final'), [c for c in fcols if c != 'adjusted_confidence'],
              'final_umap_stage1space_known.pdf', 'final', title='Final cell types (Unknown removed)')
    save_dotplot(known(adata, 'celltype_final'), mk, 'celltype_final', 'final_dotplot_known.pdf',
                 'final', title='markers by final cell type (known)')
    correlation_heatmap(known(adata, 'celltype_final'), 'celltype_final', MARKER_GENES, ref_centroids,
                        'final_corr_markers.pdf', 'final', title='final markers:')
    confidence_hist(adata, 'adjusted_confidence', 'adjusted_confidence_distribution.pdf', 'final',
                    title='adjusted confidence = s1_conf x s2_conf')

    # ── FINAL MERFISH-ALONE UMAP : re-embed the confident cells on their own ─────────────────
    print('\n[final] MERFISH-only re-embedding of the confidently annotated cells')
    known_ad = adata[adata.obs['celltype_final'].astype(str) != 'Unknown'].copy()
    print(f"  kept {known_ad.n_obs:,} / {adata.n_obs:,} cells")
    rep_f = embed(known_ad, GENES, BATCH_KEY, n_pcs_use=N_PCS, do_umap=False)
    lf, uf = cluster_umap_on_rep(known_ad.obsm[rep_f], ANNOT_RES, n_pcs=N_PCS, min_dist=MIN_DIST)
    known_ad.obs['leiden_final'] = lf
    known_ad.obsm['X_umap'] = uf
    save_umap(known_ad, ['celltype_final', 'leiden_final', 'age', 'sex'],
              'final_umap_merfishonly_known.pdf', 'final',
              title='MERFISH-only embedding of confidently annotated cells')
    save_dotplot(known_ad, mk, 'leiden_final', 'final_dotplot_merfishonly_cluster.pdf', 'final',
                 title='markers by MERFISH-only cluster')
    for key, order in [('age', ['Young', 'Old', 'VeryOld']), ('experiment', None), ('sample', None)]:
        if key in known_ad.obs and known_ad.obs[key].nunique() > 1:
            proportion_stackbar(known_ad.obs, key, 'celltype_final', f'final_proportion_by_{key}.pdf',
                                'final', order=order)

    # ── save objects ─────────────────────────────────────────────────────────────────────────
    save_object(known_ad, 'merfish_final_known.h5ad')
    save_object(adata, 'merfish_annotated.h5ad')
    print('\nDone.')


if __name__ == '__main__':
    main()
