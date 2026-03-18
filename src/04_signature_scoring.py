import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Five psoriasis-relevant signatures
PSORIASIS_SIGNATURES = {
    "keratinocyte_hyperproliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "MCM2",
        "KRT6A", "KRT16", "KRT17", "S100A7", "S100A8", "S100A9"
    ],
    "il17_il23_axis": [
        "IL17A", "IL17F", "IL17C", "IL22", "IL23A", "IL23R",
        "STAT3", "RORC", "IL36A", "IL36G", "IL36RN", "CXCL8"
    ],
    "antimicrobial_barrier_dysfunction": [
        "DEFB4A", "DEFB4B", "SERPINB4", "PI3", "SPRR1A", "SPRR2A",
        "FLG", "LOR", "IVL", "KLK5", "KLK7", "CDSN"
    ],
    "inflammatory_activation": [
        "TNF", "IL1B", "IL6", "CXCL1", "CXCL2", "CXCL10",
        "CCL20", "ICAM1", "VCAM1", "NFKB1", "NFKB2", "RELB"
    ],
    "t_cell_immune_infiltration": [
        "CD3D", "CD3E", "CD8A", "GZMB", "PRF1", "IFNG",
        "FOXP3", "IL2RA", "PDCD1", "LAG3", "TIGIT", "HAVCR2"
    ],
}

def score_signatures(adata, signatures, cluster_key="leiden"):
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    cluster_scores = {sig: [] for sig in signatures}
    for sig_name, genes in signatures.items():
        present = [g for g in genes if g in adata.var_names]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            print("  " + sig_name + ": " + str(len(present)) + "/" + str(len(genes)) + " genes found")
        else:
            print("  " + sig_name + ": all " + str(len(genes)) + " genes found")
        if not present:
            adata.obs["score_" + sig_name] = 0.0
            for _ in clusters:
                cluster_scores[sig_name].append(0.0)
            continue
        e = adata[:, present].X
        if hasattr(e, "toarray"):
            e = e.toarray()
        cell_scores = np.array(e.mean(axis=1)).flatten()
        adata.obs["score_" + sig_name] = cell_scores
        for cl in clusters:
            mask = adata.obs[cluster_key] == cl
            cluster_scores[sig_name].append(float(cell_scores[mask].mean()))
    df = pd.DataFrame(cluster_scores, index=clusters)
    df.index.name = "cluster"
    return adata, df

def score_by_condition(adata, signatures):
    """Score signatures by PP vs PN condition."""
    rows = []
    for cond in ["PP", "PN"]:
        mask = adata.obs["condition"] == cond
        row = {"condition": cond, "n_cells": int(mask.sum())}
        for sig in signatures:
            col = "score_" + sig
            row[sig] = round(float(adata.obs.loc[mask, col].mean()), 4) \
                       if col in adata.obs.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def score_by_cell_type(adata, signatures):
    rows = []
    for ct in adata.obs["cell_type"].unique():
        mask = adata.obs["cell_type"] == ct
        row = {"cell_type": ct, "n_cells": int(mask.sum())}
        for sig in signatures:
            col = "score_" + sig
            row[sig] = round(float(adata.obs.loc[mask, col].mean()), 4) \
                       if col in adata.obs.columns else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False)

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 04: Psoriasis Signature Scoring")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_annotated.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 03_annotate_clusters.py first.")
        sys.exit(1)
    print("\n[1/5] Loading annotated object...")
    adata = sc.read_h5ad(in_path)
    print("  Loaded:", adata.n_obs, "cells x", adata.n_vars, "genes")
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print("\n[2/5] Scoring psoriasis signatures...")
    adata, score_df = score_signatures(adata, PSORIASIS_SIGNATURES)
    score_df["psoriasis_primary_score"] = (
        score_df.get("keratinocyte_hyperproliferation", 0) +
        score_df.get("il17_il23_axis", 0))
    score_df["psoriasis_composite_score"] = score_df[
        [c for c in PSORIASIS_SIGNATURES if c in score_df.columns]].sum(axis=1)
    top_clusters = score_df.nlargest(3, "psoriasis_primary_score")
    print("\n  Top 3 pro-psoriasis clusters:")
    for cl, row in top_clusters.iterrows():
        ct = adata.obs.loc[adata.obs["leiden"] == str(cl), "cell_type"].values[0] \
             if "cell_type" in adata.obs.columns else "unknown"
        print("    Cluster", cl, "(" + ct + "): hyperproliferation=" +
              str(round(row.get("keratinocyte_hyperproliferation", 0), 4)) +
              ", IL17=" + str(round(row.get("il17_il23_axis", 0), 4)))
    print("\n[3/5] Scores by condition (PP vs PN)...")
    cond_scores = score_by_condition(adata, PSORIASIS_SIGNATURES)
    print(cond_scores.round(4).to_string(index=False))
    print("\n[4/5] Scores by cell type...")
    if "cell_type" in adata.obs.columns:
        ct_scores = score_by_cell_type(adata, PSORIASIS_SIGNATURES)
        print(ct_scores.head(8).round(4).to_string(index=False))
    else:
        ct_scores = pd.DataFrame()
    print("\n[5/5] Saving...")
    adata.uns["pro_psoriasis_clusters"] = top_clusters.index.tolist()
    out_path = os.path.join(PROCESSED_DIR, "adata_scored.h5ad")
    adata.write_h5ad(out_path)
    score_df.to_csv(os.path.join(PROCESSED_DIR, "signature_scores.csv"))
    cond_scores.to_csv(os.path.join(PROCESSED_DIR, "signature_scores_by_condition.csv"), index=False)
    if not ct_scores.empty:
        ct_scores.to_csv(os.path.join(PROCESSED_DIR, "signature_scores_bytype.csv"), index=False)
    print("\n" + "=" * 60)
    print("Script 04 complete. ->", out_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
