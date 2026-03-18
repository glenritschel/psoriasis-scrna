import os, sys, time, re
import numpy as np
import pandas as pd
import gseapy as gp
import scanpy as sc

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

N_TOP_GENES = 150
TOP_PER_CLUSTER = 15
ENRICHR_DELAY = 1.0
ENRICHR_LIBRARIES = [
    "LINCS_L1000_Chem_Pert_up",
    "LINCS_L1000_Chem_Pert_down",
    "GO_Biological_Process_2023",
    "Reactome_2022",
    "KEGG_2021_Human",
]

def clean_compound_name(term):
    m = re.match(r'^LJP\d+\s+\S+\s+\S+?-(.+)-[\d.]+$', term.strip())
    if m:
        return m.group(1).strip()
    parts = term.split("_")
    return parts[0].strip() if parts else term.strip()

def run_enrichr(query_id, up_genes, down_genes):
    results = []
    for direction, genes, reversal_lib in [
        ("up",   up_genes,   "LINCS_L1000_Chem_Pert_down"),
        ("down", down_genes, "LINCS_L1000_Chem_Pert_up"),
    ]:
        if not genes:
            continue
        for lib in ENRICHR_LIBRARIES:
            try:
                enr = gp.enrichr(gene_list=genes, gene_sets=lib, outdir=None, verbose=False)
                df = enr.results.copy()
                if df.empty:
                    continue
                df["query_id"] = query_id
                df["query_direction"] = direction
                df["library"] = lib
                if lib in ("LINCS_L1000_Chem_Pert_up", "LINCS_L1000_Chem_Pert_down"):
                    adj_p = df["Adjusted P-value"].clip(lower=1e-300)
                    sign = 1.0 if lib == reversal_lib else -1.0
                    df["reversal_score"] = sign * (-np.log10(adj_p))
                    df["compound"] = df["Term"].apply(clean_compound_name)
                else:
                    df["reversal_score"] = 0.0
                    df["compound"] = df["Term"]
                results.append(df)
                time.sleep(ENRICHR_DELAY)
            except Exception as e:
                print("    WARNING: Enrichr failed for", query_id, lib, ":", e)
                time.sleep(ENRICHR_DELAY * 2)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def deduplicate_and_rank(raw_df, adata=None):
    if raw_df.empty:
        return pd.DataFrame()
    lincs_mask = raw_df["library"].str.startswith("LINCS_L1000")
    lincs_df = raw_df[lincs_mask & (raw_df["reversal_score"] > 0)].copy()
    if lincs_df.empty:
        print("  WARNING: No positive LINCS reversal scores.")
        return pd.DataFrame()
    top = lincs_df.sort_values("reversal_score", ascending=False).groupby("query_id").head(TOP_PER_CLUSTER)
    agg = top.groupby("compound").agg(
        max_reversal_score=("reversal_score", "max"),
        n_queries=("query_id", "nunique"),
        queries=("query_id", lambda x: ",".join(sorted(set(x.astype(str))))),
        best_query=("query_id", lambda x: x.loc[x.index[top.loc[x.index, "reversal_score"].argmax()]]),
    ).reset_index()
    return agg.sort_values("max_reversal_score", ascending=False)

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 06: LINCS L1000 Reversal Scoring")
    print("Primary query: PP (lesional) vs PN (uninvolved) DE")
    print("Secondary: cluster-vs-rest DE")
    print("=" * 60)

    # Primary: use PP vs PN DE as the main reversal query
    pp_pn_path = os.path.join(PROCESSED_DIR, "de_PP_vs_PN.csv")
    de_path = os.path.join(PROCESSED_DIR, "de_top_genes.csv")
    if not os.path.exists(pp_pn_path):
        print("ERROR:", pp_pn_path, "not found. Run 05_differential_expression.py first.")
        sys.exit(1)
    print("\n[1/4] Loading DE gene lists...")
    pp_vs_pn = pd.read_csv(pp_pn_path)
    top_genes_df = pd.read_csv(de_path)
    n_clusters = top_genes_df["cluster"].nunique()
    print("  PP vs PN DE:", len(pp_vs_pn), "genes")
    print("  Cluster DE:", len(top_genes_df), "gene-cluster pairs across", n_clusters, "clusters")
    adata_path = os.path.join(PROCESSED_DIR, "adata_de.h5ad")
    adata = sc.read_h5ad(adata_path) if os.path.exists(adata_path) else None

    print("\n[2/4] Submitting to Enrichr...")
    all_results = []

    # Primary query: PP vs PN whole-tissue reversal
    print("  Primary query: PP vs PN lesional reversal...")
    pp_up   = pp_vs_pn[pp_vs_pn["score"] > 0].head(N_TOP_GENES)["gene"].tolist()
    pp_down = pp_vs_pn[pp_vs_pn["score"] < 0].tail(N_TOP_GENES)["gene"].tolist()
    prim_res = run_enrichr("PP_vs_PN", pp_up, pp_down)
    if not prim_res.empty:
        all_results.append(prim_res)
        print("   ", (prim_res["reversal_score"] > 0).sum(), "reversal hits")

    # Keratinocyte-specific PP vs PN if available
    kc_path = os.path.join(PROCESSED_DIR, "de_keratinocyte_PP_vs_PN.csv")
    if os.path.exists(kc_path):
        kc_de = pd.read_csv(kc_path)
        kc_up   = kc_de[kc_de["score"] > 0].head(N_TOP_GENES)["gene"].tolist()
        kc_down = kc_de[kc_de["score"] < 0].tail(N_TOP_GENES)["gene"].tolist()
        print("  Keratinocyte PP vs PN query...")
        kc_res = run_enrichr("Keratinocyte_PP_vs_PN", kc_up, kc_down)
        if not kc_res.empty:
            all_results.append(kc_res)
            print("   ", (kc_res["reversal_score"] > 0).sum(), "reversal hits")

    # Cluster-vs-rest queries
    for i, cl in enumerate(top_genes_df["cluster"].unique()):
        print("  Cluster", cl, "(" + str(i+1) + "/" + str(n_clusters) + ")...", end=" ", flush=True)
        up   = top_genes_df.loc[(top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "up"), "gene"].tolist()
        down = top_genes_df.loc[(top_genes_df["cluster"] == cl) & (top_genes_df["direction"] == "down"), "gene"].tolist()
        res = run_enrichr("cluster_" + str(cl), up, down)
        if not res.empty:
            all_results.append(res)
            print((res["reversal_score"] > 0).sum(), "reversal hits")
        else:
            print("no results")

    if not all_results:
        print("ERROR: No Enrichr results.")
        sys.exit(1)
    raw_results = pd.concat(all_results, ignore_index=True)
    raw_path = os.path.join(PROCESSED_DIR, "lincs_results_raw.csv")
    raw_results.to_csv(raw_path, index=False)
    print("  Raw results:", len(raw_results), "rows")
    print("\n[3/4] Deduplicating and ranking...")
    candidates = deduplicate_and_rank(raw_results, adata)
    if not candidates.empty:
        print(" ", len(candidates), "unique compounds identified")
        print(candidates[["compound", "max_reversal_score", "n_queries"]].head(10).round(2).to_string(index=False))
    print("\n[4/4] Saving...")
    cand_path = os.path.join(PROCESSED_DIR, "lincs_candidates.csv")
    candidates.to_csv(cand_path, index=False)
    print("\n" + "=" * 60)
    print("Script 06 complete. ->", cand_path, "(", len(candidates), "compounds )")
    print("=" * 60)

if __name__ == "__main__":
    main()
