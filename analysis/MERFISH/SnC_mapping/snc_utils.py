"""
snc_utils.py — shared functions for senescent-cell (SnC) mapping and SenSpot detection.

    load_snc_reference       reference nuclei, log-normalised, with the binary senescence label
    compute_topn_pairs       cosine top-n links from each reference nucleus to MERFISH cells
    transfer_prevalence_matched
                             P(senescent) per MERFISH cell + prevalence-matched binary call
    find_dbscan_clusters     DBSCAN of senescent cells in physical space
    grow_senspots            every cell within eps of a clustered senescent cell joins its SenSpot
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree

# ── column names / labels shared by every step ──────────────────────────────────────────────
REF_CELLTYPE_COL = 'celltype'                    # cell type on the reference
MERFISH_CELLTYPE_COL = 'celltype_final'          # cell type on MERFISH (annotation step)
REF_SNC_COL = 'predicted_senescence_label'       # binary senescence label on the reference
SNC_CLASS1 = 'senescent'
SNC_CLASS2 = 'non-senescent'
SNC_OUT_COL = 'SnC_from_sc'                      # transferred label written onto MERFISH

DBSCAN_CLUSTER_COL = 'senescent_dbscan_cluster'  # DBSCAN cluster id (senescent cells only)
SENSPOT_MEMBER_COL = 'in_senspot'                # bool: cell lies in a SenSpot
SENSPOT_ID_COL = 'senspot_cluster'               # SenSpot id for member cells, NA otherwise

# ── analysis parameters (manuscript operating point) ────────────────────────────────────────
TOP_N = 300            # MERFISH cells linked per reference nucleus
COS_THR = 0.2          # links below this cosine similarity are discarded
MIN_NEIGHBORS = 3      # MERFISH cells with fewer incoming links stay unlabelled
DBSCAN_EPS = 50        # µm; DBSCAN radius AND SenSpot growth radius
DBSCAN_MIN_SAMPLES = 5


def snc_label_cols(top_n):
    """(label column, score column) for one top_n in the per-section labels CSV."""
    return f'{SNC_OUT_COL}_topn{top_n}', f'{SNC_OUT_COL}_score_topn{top_n}'


def read_gene_list(path):
    """One gene symbol per line, no header."""
    return pd.read_csv(path, header=None)[0].astype(str).tolist()


def read_sections(path):
    """Section table written by 00_export_sections.py (columns sid, age, sex, n_cells)."""
    return pd.read_csv(path, sep='\t', dtype={'sid': str})


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Reference
# ════════════════════════════════════════════════════════════════════════════════════════════
def load_snc_reference(ref_h5ad, snc_meta_csv, sex=None, age=None):
    """
    Reference nuclei, CP10K + log1p normalised, with REF_SNC_COL joined from `snc_meta_csv`
    (indexed by the same cell ids as the reference obs_names). Nuclei absent from the CSV keep
    NaN and take no part in the transfer. Optionally restricted to one sex and age group.
    """
    print(f'Reading reference: {ref_h5ad}')
    ad_sc = sc.read_h5ad(ref_h5ad)
    sc.pp.normalize_total(ad_sc, target_sum=1e4)
    sc.pp.log1p(ad_sc)
    sc.pp.filter_genes(ad_sc, min_cells=1)
    for col in ad_sc.obs.columns:
        ad_sc.obs[col] = ad_sc.obs[col].astype(str)
    ad_sc.obs_names = ad_sc.obs_names.astype(str)

    snc = pd.read_csv(snc_meta_csv, index_col=0)
    if REF_SNC_COL not in snc.columns:
        raise KeyError(f'{REF_SNC_COL!r} not in {snc_meta_csv}')
    snc.index = snc.index.astype(str)
    ad_sc.obs = ad_sc.obs.merge(snc[[REF_SNC_COL]], left_index=True, right_index=True, how='left')

    n_annot = int(ad_sc.obs[REF_SNC_COL].notna().sum())
    print(f'  senescence label on {n_annot:,}/{ad_sc.n_obs:,} reference nuclei')
    if n_annot == 0:
        raise ValueError('No reference nucleus matched the senescence table — check that its '
                         'index uses the same cell ids as the reference obs_names.')

    if sex is not None:
        ad_sc = ad_sc[ad_sc.obs['sex'] == sex, :]
    if age is not None:
        ad_sc = ad_sc[ad_sc.obs['age'] == age, :]
    return ad_sc


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Label transfer
# ════════════════════════════════════════════════════════════════════════════════════════════
def compute_topn_pairs(ad_sc, ad_sp, genes, top_n=TOP_N, cos_thr=COS_THR):
    """
    For each reference nucleus, its `top_n` most cosine-similar MERFISH cells over `genes`
    (expression taken from .X of both objects). Links with similarity < cos_thr are dropped.

    Returns a long table with columns sc_cell, sp_cell, cosine_sim.
    """
    genes = pd.Index(genes).intersection(ad_sc.var_names).intersection(ad_sp.var_names)
    if len(genes) == 0:
        raise ValueError('No overlapping genes between the reference and MERFISH.')

    X_sc = ad_sc[:, genes].X
    X_sp = ad_sp[:, genes].X
    X_sc = np.asarray(X_sc.toarray() if hasattr(X_sc, 'toarray') else X_sc, dtype=np.float32)
    X_sp = np.asarray(X_sp.toarray() if hasattr(X_sp, 'toarray') else X_sp, dtype=np.float32)
    n_sc, n_sp = X_sc.shape[0], X_sp.shape[0]

    sim = cosine_similarity(X_sc, X_sp)                         # n_sc x n_sp

    k = min(top_n, n_sp)
    top_unsorted = np.argpartition(sim, -k, axis=1)[:, -k:]     # k largest per row, unordered
    rows = np.arange(n_sc)[:, None]
    sims_unsorted = sim[rows, top_unsorted]
    order = np.argsort(-sims_unsorted, axis=1)
    top_idx = top_unsorted[rows, order]
    top_sims = sims_unsorted[rows, order]

    pairs = pd.DataFrame({
        'sc_cell': np.asarray(ad_sc.obs_names)[np.repeat(np.arange(n_sc), k)],
        'sp_cell': np.asarray(ad_sp.obs_names)[top_idx.ravel()],
        'cosine_sim': top_sims.ravel(),
    })
    if cos_thr is not None:
        pairs = pairs[pairs['cosine_sim'] >= float(cos_thr)].reset_index(drop=True)
    return pairs


def transfer_prevalence_matched(ad_sc, ad_sp, pairs, annot_col=REF_SNC_COL,
                                class1=SNC_CLASS1, class2=SNC_CLASS2,
                                min_neighbors=MIN_NEIGHBORS):
    """
    Binary label transfer by prevalence matching.

      1. For each MERFISH cell, sum the cosine similarity of its incoming links from class1
         (w_1) and class2 (w_2) reference nuclei; P(class1) = w_1 / (w_1 + w_2).
      2. Cells with < min_neighbors links are left unlabelled (NaN).
      3. Among the remaining N cells, the top k = round(prior_1 * N) by P(class1) (ties broken
         by total link weight) are called class1, the rest class2, where prior_1 is the class1
         fraction among the reference nuclei passed in.

    Returns (labels, scores, details):
      labels   Series over ad_sp.obs_names — class1 / class2 / NaN
      scores   Series over ad_sp.obs_names — P(class1); NaN for cells with no links
      details  per-linked-cell table: w_1, w_2, total_w, prob1, n_neighbors, eligible,
               assigned_label, prior, k_assigned, N_pool
    """
    empty = (pd.Series(np.nan, index=ad_sp.obs_names, dtype=object),
             pd.Series(np.nan, index=ad_sp.obs_names, dtype=float), pd.DataFrame())

    sc_labels = ad_sc.obs[annot_col].dropna()
    uniq = set(sc_labels.unique())
    if uniq != {class1, class2}:
        raise ValueError(f'Expected reference classes {{{class1!r}, {class2!r}}}, found {sorted(uniq)}')
    prior_1 = float(sc_labels.value_counts(normalize=True).get(class1, 0.0))

    df = pairs.merge(ad_sc.obs[[annot_col]], left_on='sc_cell', right_index=True,
                     how='left').dropna(subset=[annot_col])
    if df.empty:
        return empty

    w_by = df.groupby(['sp_cell', annot_col])['cosine_sim'].sum().unstack(fill_value=0.0)
    for c in (class1, class2):
        if c not in w_by.columns:
            w_by[c] = 0.0
    w_by = w_by[[class1, class2]]
    n_neighbors = df.groupby('sp_cell').size().reindex(w_by.index).fillna(0).astype(int)
    total_w = (w_by[class1] + w_by[class2]).astype(float)
    prob1 = (w_by[class1] / total_w.replace(0.0, np.nan)).fillna(0.0)

    res = pd.DataFrame({
        'w_1': w_by[class1].values, 'w_2': w_by[class2].values, 'total_w': total_w.values,
        'prob1': prob1.values, 'n_neighbors': n_neighbors.values,
    }, index=pd.Index(w_by.index, name='sp_cell'))

    eligible = res['n_neighbors'] >= int(min_neighbors)
    pool = res.loc[eligible]
    N = pool.shape[0]
    assigned = pd.Series(np.nan, index=res.index, dtype=object)
    k = 0
    if N > 0:
        k = max(0, min(int(round(prior_1 * N)), N))
        pool = pool.sort_values(['prob1', 'total_w'], ascending=[False, False])
        assigned.loc[pool.index[:k]] = class1
        assigned.loc[pool.index[k:]] = class2
    assigned.loc[~eligible] = np.nan

    res['eligible'] = eligible.values
    res['assigned_label'] = assigned.values
    res['prior'] = prior_1
    res['k_assigned'] = k
    res['N_pool'] = N

    labels = assigned.reindex(ad_sp.obs_names)
    scores = prob1.reindex(ad_sp.obs_names).astype(float)
    return labels, scores, res


# ════════════════════════════════════════════════════════════════════════════════════════════
#  Spatial clustering
# ════════════════════════════════════════════════════════════════════════════════════════════
def find_dbscan_clusters(obs, xy, label_col, label=SNC_CLASS1, eps=DBSCAN_EPS,
                         min_samples=DBSCAN_MIN_SAMPLES, out_col=DBSCAN_CLUSTER_COL):
    """
    DBSCAN on the cells whose `label_col` == `label`, in physical coordinates `xy` (µm).
    Writes obs[out_col]: cluster id (string) for clustered cells; NA for DBSCAN noise
    (isolated senescent cells) and for every other cell.
    """
    is_lab = obs[label_col].astype('string').eq(label).fillna(False).to_numpy()
    out = pd.Series(pd.NA, index=obs.index, dtype='string')
    if is_lab.any():
        db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy[is_lab])
        out[is_lab] = pd.Series(db, index=obs.index[is_lab]).astype(int).astype('string')
        out = out.replace('-1', pd.NA)
    obs[out_col] = pd.Categorical(out)
    return obs


def grow_senspots(obs, xy, cluster_col=DBSCAN_CLUSTER_COL, eps=DBSCAN_EPS):
    """
    Grow each DBSCAN cluster into a SenSpot: every cell (any type, labelled or not) whose
    nearest clustered senescent cell lies within `eps` joins that cell's SenSpot.
    Writes obs[SENSPOT_MEMBER_COL] (bool) and obs[SENSPOT_ID_COL] (SenSpot id or NA).
    """
    clusters = obs[cluster_col].astype('string')
    is_core = clusters.notna().to_numpy()
    if not is_core.any():
        obs[SENSPOT_MEMBER_COL] = False
        obs[SENSPOT_ID_COL] = pd.Categorical([pd.NA] * len(obs))
        return obs

    dist, idx = BallTree(xy[is_core]).query(xy, k=1)
    dist, idx = dist.ravel(), idx.ravel()
    nearest = clusters[is_core].to_numpy()[idx]
    member = dist < float(eps)

    sid = pd.Series(nearest, index=obs.index, dtype=object)
    sid[~member] = pd.NA
    obs[SENSPOT_MEMBER_COL] = member
    obs[SENSPOT_ID_COL] = pd.Categorical(sid)
    return obs
