import os, sys
import numpy as np
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

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
CONFIDENCE_THRESHOLD = 1.2

def score_clusters(adata, marker_dict, cluster_key="leiden"):
    clusters = sorted(adata.obs[cluster_key].unique(), key=lambda x: int(x))
    scores = {ct: [] for ct in marker_dict}
    for cl in clusters:
        mask = adata.obs[cluster_key] == cl
        for ct, markers in marker_dict.items():
            present = [m for m in markers if m in adata.var_names]
            if not present:
                scores[ct].append(0.0)
                continue
            e = adata[mask, present].X
            if hasattr(e, "toarray"):
                e = e.toarray()
            scores[ct].append(float(e.mean()))
    df = pd.DataFrame(scores, index=clusters)
    df.index.name = "cluster"
    return df

def assign_annotations(score_df):
    rows = []
    for cluster in score_df.index:
        row = score_df.loc[cluster]
        best = row.idxmax()
        best_score = row.max()
        sorted_s = row.sort_values(ascending=False)
        runner_up = sorted_s.iloc[1] if len(sorted_s) > 1 else 0.0
        if best_score == 0.0:
            label, conf = "unknown", "low"
        elif runner_up == 0.0 or best_score >= CONFIDENCE_THRESHOLD * runner_up:
            label, conf = best, "high"
        else:
            label, conf = best + "_mixed", "low"
        rows.append({
            "cluster": cluster, "annotation": label, "confidence": conf,
            "best_score": round(best_score, 4), "runner_up_score": round(runner_up, 4),
        })
    return pd.DataFrame(rows)

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 03: Cell Type Annotation")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_scvi.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 02_scvi_embed.py first.")
        sys.exit(1)
    print("\n[1/4] Loading scVI object...")
    adata = sc.read_h5ad(in_path)
    print("  Loaded:", adata.n_obs, "cells,", adata.obs["leiden"].nunique(), "clusters")
    if "norm_log" not in adata.layers:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    print("\n[2/4] Scoring clusters against cell type markers...")
    score_df = score_clusters(adata, CELL_TYPE_MARKERS)
    print("\n[3/4] Assigning annotations...")
    ann_df = assign_annotations(score_df)
    counts = adata.obs["leiden"].value_counts().rename("n_cells")
    summary = ann_df.set_index("cluster").join(counts).sort_values("n_cells", ascending=False)
    print("\n  Cluster annotations:")
    print("  " + "-" * 70)
    for cl, row in summary.iterrows():
        print("  Cluster", cl, "|", row["annotation"], "|",
              row["confidence"], "|", int(row["n_cells"]), "cells")
    # Keratinocyte cluster summary
    kc_clusters = ann_df[ann_df["annotation"].str.startswith("keratinocyte")]
    print("\n  Keratinocyte clusters:", len(kc_clusters),
          "->", kc_clusters["annotation"].value_counts().to_dict())
    print("\n[4/4] Saving...")
    adata.obs["cell_type"] = adata.obs["leiden"].map(
        dict(zip(ann_df["cluster"].astype(str), ann_df["annotation"])))
    adata.obs["annotation_confidence"] = adata.obs["leiden"].map(
        dict(zip(ann_df["cluster"].astype(str), ann_df["confidence"])))
    adata.uns["cell_type_scores"] = score_df.to_dict()
    out_path = os.path.join(PROCESSED_DIR, "adata_annotated.h5ad")
    adata.write_h5ad(out_path)
    ann_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_annotations.csv"), index=False)
    score_df.to_csv(os.path.join(PROCESSED_DIR, "cluster_marker_scores.csv"))
    print("\n" + "=" * 60)
    print("Script 03 complete. ->", out_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
