import os, sys
import pandas as pd
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
N_TOP_GENES = 150

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 05: Differential Expression")
    print("PP (lesional) vs PN (uninvolved) within each cluster")
    print("=" * 60)
    in_path = os.path.join(PROCESSED_DIR, "adata_scored.h5ad")
    if not os.path.exists(in_path):
        print("ERROR:", in_path, "not found. Run 04_signature_scoring.py first.")
        sys.exit(1)
    print("\n[1/5] Loading scored object...")
    adata = sc.read_h5ad(in_path)
    n_clusters = adata.obs["leiden"].nunique()
    print("  Loaded:", adata.n_obs, "cells,", n_clusters, "clusters")
    print("  PP:", (adata.obs["condition"] == "PP").sum(),
          "| PN:", (adata.obs["condition"] == "PN").sum())
    if "norm_log" in adata.layers:
        adata.X = adata.layers["norm_log"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Primary DE: PP vs PN across all cells (disease vs uninvolved)
    print("\n[2/5] Running PP vs PN Wilcoxon DE (all cells)...")
    sc.tl.rank_genes_groups(adata, groupby="condition", groups=["PP"],
                            reference="PN", method="wilcoxon",
                            use_raw=False, key_added="rank_genes_PPvPN", pts=True)
    result = adata.uns["rank_genes_PPvPN"]
    pp_vs_pn = pd.DataFrame({
        "gene": result["names"]["PP"],
        "score": result["scores"]["PP"],
        "pval_adj": result["pvals_adj"]["PP"],
        "log2fc": result["logfoldchanges"]["PP"],
    }).sort_values("score", ascending=False)
    pp_vs_pn_path = os.path.join(PROCESSED_DIR, "de_PP_vs_PN.csv")
    pp_vs_pn.to_csv(pp_vs_pn_path, index=False)
    print("  PP vs PN DE: top upregulated:", pp_vs_pn.head(5)["gene"].tolist())
    print("  PP vs PN DE: top downregulated:", pp_vs_pn.tail(5)["gene"].tolist())

    # Secondary DE: cluster vs rest (for LINCS input)
    print("\n[3/5] Running cluster-vs-rest Wilcoxon DE...")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            use_raw=False, key_added="rank_genes_groups", pts=True)
    result2 = adata.uns["rank_genes_groups"]
    rows = []
    for cl in result2["names"].dtype.names:
        genes  = result2["names"][cl]
        scores = result2["scores"][cl]
        pvals  = result2["pvals_adj"][cl]
        df_cl = pd.DataFrame({"cluster": cl, "gene": genes, "score": scores, "pval_adj": pvals})
        top_up   = df_cl.nlargest(N_TOP_GENES, "score").copy()
        top_up["direction"] = "up"
        top_down = df_cl.nsmallest(N_TOP_GENES, "score").copy()
        top_down["direction"] = "down"
        rows.extend([top_up, top_down])
    top_genes_df = pd.concat(rows, ignore_index=True)
    print("  Extracted", len(top_genes_df), "gene-cluster pairs")

    # Focused DE: PP vs PN within keratinocyte clusters specifically
    print("\n[4/5] Running PP vs PN DE within keratinocyte clusters...")
    kc_mask = adata.obs["cell_type"].str.startswith("keratinocyte") if "cell_type" in adata.obs.columns else pd.Series(True, index=adata.obs_names)
    adata_kc = adata[kc_mask].copy()
    if adata_kc.n_obs > 100 and (adata_kc.obs["condition"] == "PP").sum() > 50:
        sc.tl.rank_genes_groups(adata_kc, groupby="condition", groups=["PP"],
                                reference="PN", method="wilcoxon",
                                use_raw=False, key_added="rank_kc_PPvPN")
        kc_result = adata_kc.uns["rank_kc_PPvPN"]
        kc_de = pd.DataFrame({
            "gene": kc_result["names"]["PP"],
            "score": kc_result["scores"]["PP"],
            "pval_adj": kc_result["pvals_adj"]["PP"],
        }).sort_values("score", ascending=False)
        kc_de.to_csv(os.path.join(PROCESSED_DIR, "de_keratinocyte_PP_vs_PN.csv"), index=False)
        print("  Keratinocyte DE: top genes:", kc_de.head(5)["gene"].tolist())
    else:
        print("  Insufficient keratinocyte cells for focused DE.")

    # Pro-psoriasis focused DE
    pro_clusters = list(adata.uns.get("pro_psoriasis_clusters", []))
    if pro_clusters:
        pro_str = [str(c) for c in pro_clusters]
        adata.obs["pro_pso_group"] = adata.obs["leiden"].apply(
            lambda x: "pro_psoriasis" if str(x) in pro_str else "other")
        sc.tl.rank_genes_groups(adata, groupby="pro_pso_group",
                                groups=["pro_psoriasis"], reference="other",
                                method="wilcoxon", use_raw=False,
                                key_added="rank_genes_propso")
        pr = adata.uns["rank_genes_propso"]
        pr_df = pd.DataFrame({
            "gene": pr["names"]["pro_psoriasis"],
            "score": pr["scores"]["pro_psoriasis"],
            "pval_adj": pr["pvals_adj"]["pro_psoriasis"],
        }).sort_values("score", ascending=False)
        pr_df.to_csv(os.path.join(PROCESSED_DIR, "de_propsoriasis_vs_rest.csv"), index=False)
        print("  Pro-psoriasis clusters DE saved.")

    print("\n[5/5] Saving...")
    de_path = os.path.join(PROCESSED_DIR, "de_top_genes.csv")
    top_genes_df.to_csv(de_path, index=False)
    adata.write_h5ad(os.path.join(PROCESSED_DIR, "adata_de.h5ad"))
    print("\n" + "=" * 60)
    print("Script 05 complete.")
    print("  PP vs PN DE:", pp_vs_pn_path)
    print("  Cluster DE:", de_path)
    print("=" * 60)

if __name__ == "__main__":
    main()
