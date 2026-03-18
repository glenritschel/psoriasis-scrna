import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SCVI_PARAMS = {"n_latent": 30, "n_layers": 2, "n_hidden": 128}
SCVI_TRAIN_PARAMS = {"max_epochs": 400, "early_stopping": False}
N_NEIGHBORS = 15
N_PCS = 30
RANDOM_SEED = 0
LEIDEN_RESOLUTIONS = [0.5, 0.8, 1.2]

# Marker panels for psoriasis cell type annotation
CELL_TYPE_MARKERS = {
    "keratinocyte_basal":    ["KRT5", "KRT14", "COL17A1", "ITGA6", "TP63"],
    "keratinocyte_spinous":  ["KRT1", "KRT10", "KRTDAP", "DSC1", "DSG1"],
    "keratinocyte_granular": ["FLG", "LOR", "SPRR1A", "SPRR2A", "IVL"],
    "keratinocyte_psoriatic":["KRT6A", "KRT16", "KRT17", "S100A7", "S100A8", "SERPINB4"],
    "fibroblast":            ["DCN", "LUM", "COL1A1", "PDGFRA", "THY1"],
    "t_cell":                ["CD3D", "CD3E", "CD8A", "CD4", "TRAC"],
    "myeloid":               ["CD68", "LYZ", "S100A9", "CD14", "FCGR3A"],
    "endothelial":           ["VWF", "PECAM1", "CDH5", "CLDN5"],
    "melanocyte":            ["MLANA", "DCT", "TYRP1", "PMEL"],
    "mast_cell":             ["TPSAB1", "TPSB2", "CPA3", "KIT"],
}

def set_seeds(seed=0):
    import torch, random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def train_scvi(adata):
    import scvi
    scvi.settings.seed = RANDOM_SEED
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="sample")
    model = scvi.model.SCVI(adata, **SCVI_PARAMS)
    print("  Training scVI on", adata.n_obs, "cells x",
          adata.var["highly_variable"].sum(), "HVGs...")
    model.train(**SCVI_TRAIN_PARAMS, accelerator="auto")
    final_loss = model.history["train_loss_epoch"].values[-1]
    print("  Training complete. Final loss:", float(np.array(final_loss).flat[0]))
    adata.obsm["X_scVI"] = model.get_latent_representation()
    print("  Latent shape:", adata.obsm["X_scVI"].shape)
    return adata, model

def build_neighbor_graph(adata):
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.umap(adata)
    print("  Neighbour graph built")
    return adata

def run_leiden_multi_resolution(adata):
    results = {}
    for res in LEIDEN_RESOLUTIONS:
        key = "leiden_" + str(res)
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=RANDOM_SEED,
                     flavor="igraph", n_iterations=2, directed=False)
        n = adata.obs[key].nunique()
        print("  Resolution", res, ":", n, "clusters ->", key)
        results[res] = {"key": key, "n_clusters": n}
    return adata, results

def score_resolution(adata, resolution_results):
    rows = []
    for res, info in resolution_results.items():
        key = info["key"]
        best_clusters = {}
        for ct, markers in CELL_TYPE_MARKERS.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                continue
            cluster_means = {}
            for cl in adata.obs[key].unique():
                mask = adata.obs[key] == cl
                e = adata[mask, present].X
                if hasattr(e, "toarray"):
                    e = e.toarray()
                cluster_means[cl] = float(e.mean())
            best_clusters[ct] = max(cluster_means, key=cluster_means.get)
        n_distinct = len(set(best_clusters.values()))
        n_resolved = len(best_clusters)
        rows.append({
            "resolution": res,
            "n_clusters": info["n_clusters"],
            "n_celltypes_resolved": n_resolved,
            "n_distinct_best_clusters": n_distinct,
            "separation_score": n_distinct / max(n_resolved, 1),
        })
    df = pd.DataFrame(rows).sort_values("separation_score", ascending=False)
    best = df.iloc[0]
    recommended = best["resolution"]
    print("\n  Resolution comparison:")
    print(df.to_string(index=False))
    print("\n  Recommended:", recommended,
          "(score:", round(float(best["separation_score"]), 2),
          ",", int(best["n_clusters"]), "clusters)")
    return df, recommended

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 02: scVI + Leiden Clustering")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_qc.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 01_load_qc.py first.")
        sys.exit(1)
    print("\n[1/5] Loading QC object...")
    adata = sc.read_h5ad(in_path)
    print("  Loaded:", adata.n_obs, "cells x", adata.n_vars, "genes")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    print("  HVG subset:", adata_hvg.n_obs, "x", adata_hvg.n_vars)
    print("\n[2/5] Training scVI...")
    set_seeds(RANDOM_SEED)
    adata_hvg, model = train_scvi(adata_hvg)
    print("\n[3/5] Building neighbour graph and UMAP...")
    adata_hvg = build_neighbor_graph(adata_hvg)
    print("\n[4/5] Running Leiden at multiple resolutions...")
    adata_hvg, res_results = run_leiden_multi_resolution(adata_hvg)
    print("\n[5/5] Selecting best resolution...")
    res_metrics, recommended = score_resolution(adata_hvg, res_results)
    best_key = "leiden_" + str(recommended)
    adata_hvg.obs["leiden"] = adata_hvg.obs[best_key].copy()
    adata_hvg.uns["recommended_leiden_resolution"] = recommended
    adata_hvg.uns["n_leiden_clusters"] = adata_hvg.obs["leiden"].nunique()
    adata_hvg.uns["pro_psoriasis_clusters"] = []
    out_path = os.path.join(PROCESSED_DIR, "adata_scvi.h5ad")
    adata_hvg.write_h5ad(out_path)
    res_metrics.to_csv(os.path.join(PROCESSED_DIR, "resolution_metrics.csv"), index=False)
    # Condition distribution per cluster
    cond_dist = adata_hvg.obs.groupby(["leiden", "condition"]).size().unstack(fill_value=0)
    cond_dist.to_csv(os.path.join(PROCESSED_DIR, "cluster_condition_distribution.csv"))
    print("\n" + "=" * 60)
    print("Script 02 complete.")
    print("  scVI object:", out_path)
    print("  Final clusters:", adata_hvg.obs["leiden"].nunique())
    print("=" * 60)

if __name__ == "__main__":
    main()
