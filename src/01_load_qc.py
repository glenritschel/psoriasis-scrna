import os, sys, glob
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "GSE173706")
os.makedirs(PROCESSED_DIR, exist_ok=True)

QC_MIN_GENES = 200
QC_MIN_CELLS = 3
QC_MAX_MITO_PCT = 20.0
N_HVG = 4000
N_DEV_SAMPLES = None  # None = all 12; set to 2 for dev run

# GSE173706: 6 lesional (PP) + 6 uninvolved (PN) from 6 patients
# GSM IDs and condition labels
SAMPLES = [
    ("GSM5284973", "PP", "P1"),
    ("GSM5284974", "PN", "P1"),
    ("GSM5284975", "PP", "P2"),
    ("GSM5284976", "PN", "P2"),
    ("GSM5284977", "PP", "P3"),
    ("GSM5284978", "PN", "P3"),
    ("GSM5284979", "PP", "P4"),
    ("GSM5284980", "PN", "P4"),
    ("GSM5284981", "PP", "P5"),
    ("GSM5284982", "PN", "P5"),
    ("GSM5284983", "PP", "P6"),
    ("GSM5284984", "PN", "P6"),
]

def load_mtx_sample(gsm_dir, sample_id, condition, patient):
    """Load 10x MTX format sample from GSE173706."""
    # Try both possible MTX directory structures
    mtx_path = gsm_dir
    barcodes = os.path.join(mtx_path, "barcodes.tsv.gz")
    if not os.path.exists(barcodes):
        barcodes = os.path.join(mtx_path, "barcodes.tsv")
    features = os.path.join(mtx_path, "features.tsv.gz")
    if not os.path.exists(features):
        features = os.path.join(mtx_path, "genes.tsv.gz")
    if not os.path.exists(features):
        features = os.path.join(mtx_path, "genes.tsv")
    matrix = os.path.join(mtx_path, "matrix.mtx.gz")
    if not os.path.exists(matrix):
        matrix = os.path.join(mtx_path, "matrix.mtx")
    if not os.path.exists(matrix):
        raise FileNotFoundError("No matrix.mtx found in " + mtx_path)
    adata = sc.read_10x_mtx(mtx_path, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=1)
    adata.obs["sample"] = sample_id
    adata.obs["condition"] = condition   # PP = lesional, PN = uninvolved
    adata.obs["patient"] = patient
    adata.obs_names = [sample_id + "_" + bc for bc in adata.obs_names]
    return adata

def find_sample_dirs(raw_dir, samples, n_samples=None):
    """Find MTX directories for each GSM."""
    found = []
    s = samples[:n_samples] if n_samples else samples
    for gsm, condition, patient in s:
        # Look for directory named by GSM
        gsm_dir = os.path.join(raw_dir, gsm)
        if os.path.isdir(gsm_dir):
            found.append((gsm, condition, patient, gsm_dir))
        else:
            print("  WARNING: Directory not found for", gsm)
    return found

def apply_qc(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=QC_MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=QC_MIN_CELLS)
    adata = adata[adata.obs["pct_counts_mt"] < QC_MAX_MITO_PCT].copy()
    print("  QC:", n_before, "->", adata.n_obs, "cells")
    return adata

def select_hvg(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["norm_log"] = adata.X.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3",
                                 layer="counts", batch_key="sample", subset=False)
    print("  Selected", adata.var["highly_variable"].sum(), "HVGs")
    return adata

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 01: Load, QC, Condition Labelling")
    print("Dataset: GSE173706 (Ma et al. 2023)")
    print("6 patients x 2 conditions (PP lesional / PN uninvolved)")
    print("=" * 60)
    print("\n[1/4] Finding sample directories...")
    sample_dirs = find_sample_dirs(RAW_DIR, SAMPLES, n_samples=N_DEV_SAMPLES)
    print("  Found", len(sample_dirs), "of", len(SAMPLES), "sample directories")
    if not sample_dirs:
        print("ERROR: No sample directories found. Run the download cell first.")
        sys.exit(1)
    print("\n[2/4] Loading MTX samples...")
    adatas = []
    for gsm, condition, patient, gsm_dir in sample_dirs:
        sid = gsm + "_" + condition
        print("  Loading", sid, "...", end=" ", flush=True)
        try:
            a = load_mtx_sample(gsm_dir, sid, condition, patient)
            adatas.append(a)
            print(a.n_obs, "cells x", a.n_vars, "genes")
        except Exception as e:
            print("ERROR:", e)
    if not adatas:
        print("ERROR: No samples loaded.")
        sys.exit(1)
    print("\n  Concatenating", len(adatas), "samples...")
    adata = sc.concat(adatas, join="outer", fill_value=0)
    adata.layers["counts"] = (sp.csr_matrix(adata.X)
                               if not sp.issparse(adata.X) else adata.X.copy())
    print("  Combined:", adata.n_obs, "cells x", adata.n_vars, "genes")
    n_pp = (adata.obs["condition"] == "PP").sum()
    n_pn = (adata.obs["condition"] == "PN").sum()
    print("  PP (lesional):", n_pp, "| PN (uninvolved):", n_pn)
    print("\n[3/4] QC filtering...")
    adata_qc = apply_qc(adata)
    adata_qc.write_h5ad(os.path.join(PROCESSED_DIR, "psoriasis_qc.h5ad"))
    summary = adata_qc.obs.groupby(["condition", "patient"]).size().reset_index(name="n_cells")
    summary.to_csv(os.path.join(PROCESSED_DIR, "qc_summary.csv"), index=False)
    print("  QC summary:")
    print(summary.to_string(index=False))
    print("\n[4/4] Selecting HVGs...")
    adata_qc = select_hvg(adata_qc)
    out_path = os.path.join(PROCESSED_DIR, "adata_qc.h5ad")
    adata_qc.write_h5ad(out_path)
    print("\n" + "=" * 60)
    print("Script 01 complete.")
    print("  QC object:", adata_qc.n_obs, "cells ->", out_path)
    print("  PP:", (adata_qc.obs["condition"] == "PP").sum(),
          "| PN:", (adata_qc.obs["condition"] == "PN").sum())
    print("=" * 60)

if __name__ == "__main__":
    main()
