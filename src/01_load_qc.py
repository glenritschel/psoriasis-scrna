import os, sys, glob, re
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "GSE173706")
MAPPING_PATH = os.path.join(RAW_DIR, "ensembl_to_symbol.csv")
os.makedirs(PROCESSED_DIR, exist_ok=True)

QC_MIN_GENES = 200
QC_MIN_CELLS = 3
QC_MAX_MITO_PCT = 20.0
N_HVG = 4000
# Use PP and PN only (paired design); NS = normal skin also available
USE_CONDITIONS = ["PP", "PN"]  # set to ["PP", "PN", "NS"] to include normal skin

def parse_filename(fname):
    """Extract GSM, condition and patient ID from filename.
    e.g. GSM5277189_PP-30696.csv.gz -> (GSM5277189, PP, 30696)
    """
    m = re.match(r"(GSM\d+)_(NS|PN|PP)-(.+)\.csv\.gz$", fname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None

def load_csv_sample(csv_path, sample_id, condition, patient, ensg_to_symbol):
    """Load genes x cells CSV, transpose, map Ensembl IDs to symbols."""
    df = pd.read_csv(csv_path, index_col=0)
    # df is genes x cells — transpose to cells x genes
    df = df.T
    # Map Ensembl IDs to gene symbols
    df.columns = df.columns.map(lambda g: ensg_to_symbol.get(g, g))
    # Remove empty symbol mappings and deduplicate
    df = df.loc[:, df.columns != ""]
    df = df.loc[:, ~df.columns.duplicated()]
    # Build AnnData
    adata = sc.AnnData(X=sp.csr_matrix(df.values, dtype=np.float32))
    adata.obs_names = [sample_id + "_" + bc for bc in df.index]
    adata.var_names = df.columns.tolist()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_counts=1)
    adata.obs["sample"]    = sample_id
    adata.obs["condition"] = condition
    adata.obs["patient"]   = patient
    return adata

def apply_qc(adata):
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
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
    print("Dataset: GSE173706 — CSV format, Ensembl IDs")
    print("Conditions:", USE_CONDITIONS)
    print("=" * 60)

    print("\n[1/5] Loading Ensembl -> symbol mapping...")
    if not os.path.exists(MAPPING_PATH):
        print("ERROR: Mapping file not found at", MAPPING_PATH)
        print("Run the pybiomart download cell in the notebook first.")
        sys.exit(1)
    mapping_df = pd.read_csv(MAPPING_PATH)
    # Columns: 'Gene stable ID', 'HGNC symbol'
    ensg_col = [c for c in mapping_df.columns if "stable" in c.lower() or "ensembl" in c.lower()][0]
    sym_col  = [c for c in mapping_df.columns if "symbol" in c.lower() or "hgnc" in c.lower()][0]
    ensg_to_symbol = dict(zip(mapping_df[ensg_col], mapping_df[sym_col]))
    # Remove empty mappings
    ensg_to_symbol = {k: v for k, v in ensg_to_symbol.items()
                      if isinstance(v, str) and v.strip() != ""}
    print("  Loaded", len(ensg_to_symbol), "Ensembl->symbol mappings")

    print("\n[2/5] Finding CSV files...")
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "GSM*.csv.gz")))
    samples = []
    for f in csv_files:
        fname = os.path.basename(f)
        gsm, condition, patient = parse_filename(fname)
        if gsm and condition in USE_CONDITIONS:
            samples.append((gsm, condition, patient, f))
    print("  Found", len(samples), "files matching conditions", USE_CONDITIONS)
    if not samples:
        print("ERROR: No CSV files found.")
        sys.exit(1)

    print("\n[3/5] Loading CSV samples...")
    adatas = []
    for gsm, condition, patient, csv_path in samples:
        sid = gsm + "_" + condition + "_" + patient
        print("  Loading", sid, "...", end=" ", flush=True)
        try:
            a = load_csv_sample(csv_path, sid, condition, patient, ensg_to_symbol)
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
    for cond in USE_CONDITIONS:
        n = (adata.obs["condition"] == cond).sum()
        print("  " + cond + ":", n, "cells")

    print("\n[4/5] QC filtering...")
    adata_qc = apply_qc(adata)
    adata_qc.write_h5ad(os.path.join(PROCESSED_DIR, "psoriasis_qc.h5ad"))
    summary = adata_qc.obs.groupby(["condition", "patient"]).size().reset_index(name="n_cells")
    summary.to_csv(os.path.join(PROCESSED_DIR, "qc_summary.csv"), index=False)
    print("  QC summary:")
    print(summary.to_string(index=False))

    print("\n[5/5] Selecting HVGs...")
    adata_qc = select_hvg(adata_qc)
    out_path = os.path.join(PROCESSED_DIR, "adata_qc.h5ad")
    adata_qc.write_h5ad(out_path)
    print("\n" + "=" * 60)
    print("Script 01 complete.")
    print("  QC object:", adata_qc.n_obs, "cells ->", out_path)
    for cond in USE_CONDITIONS:
        n = (adata_qc.obs["condition"] == cond).sum()
        print("  " + cond + ":", n)
    print("=" * 60)

if __name__ == "__main__":
    main()
