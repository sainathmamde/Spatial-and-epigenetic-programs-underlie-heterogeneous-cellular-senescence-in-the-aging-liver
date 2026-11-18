#!/usr/bin/env python
"""Apply cell-level QC to MERFISH AnnData containing only wild-type samples.

Sample metadata and cell QC metrics are supplied in the input object.
"""

from pathlib import Path

import numpy as np
import scanpy as sc

MERFISH_PATH = '/path/to/merfish_wildtype.h5ad'
OUT_DIR = '/path/to/output/QC'

MIN_TRANSCRIPTS = 20
MIN_GENES = 10
MIN_VOLUME = 100  # µm³
MAX_VOLUME = 9000  # µm³
MAX_ANISOTROPY = 4.0
MIN_SOLIDITY = 0.50
MAX_BLANK_RATIO = 0.10


def main():
    if any(path.startswith('/path/to/') for path in (MERFISH_PATH, OUT_DIR)):
        raise SystemExit('Set MERFISH_PATH and OUT_DIR at the top of this script.')

    adata = sc.read_h5ad(MERFISH_PATH)
    obs = adata.obs
    blank_counts = np.asarray(adata.obsm['blank_genes'].sum(axis=1)).ravel()
    blank_ratio = blank_counts / (obs['transcript_count'].to_numpy() + blank_counts + 1e-9)

    failed = (
        (obs['transcript_count'].to_numpy() < MIN_TRANSCRIPTS)
        | (obs['n_genes_by_counts'].to_numpy() < MIN_GENES)
        | (obs['volume'].to_numpy() < MIN_VOLUME)
        | (obs['volume'].to_numpy() > MAX_VOLUME)
        | (obs['anisotropy'].to_numpy() > MAX_ANISOTROPY)
        | (obs['solidity'].to_numpy() < MIN_SOLIDITY)
        | (blank_ratio > MAX_BLANK_RATIO)
    )
    filtered = adata[~failed].copy()
    if 'counts' not in filtered.layers:
        filtered.layers['counts'] = filtered.X.copy()

    output = Path(OUT_DIR) / 'merfish_qc_filtered.h5ad'
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_h5ad(output)
    print(f'Kept {filtered.n_obs:,} of {adata.n_obs:,} cells. Saved to {output}')


if __name__ == '__main__':
    main()
