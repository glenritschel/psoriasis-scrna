import os, sys, time
import numpy as np
import pandas as pd
import requests

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_DELAY = 0.4
NCBI_EMAIL = "glen.ritschel@ritschelresearch.com"
NOVELTY_WEIGHTS = {"NOVEL_ALL": 3.0, "NOVEL_PSORIASIS": 1.5, "KNOWN": 1.0}
PATENT_WATCH_MIN_REVERSAL = 20.0
PATENT_WATCH_MIN_QUERIES = 2

MOA_REFERENCE = {
    "pd-0325901": "MEK1/2 inhibitor",
    "selumetinib": "MEK1/2 inhibitor",
    "trametinib": "MEK1/2 inhibitor",
    "pd-184352": "MEK1/2 inhibitor",
    "azd-8330": "MEK1/2 inhibitor",
    "ldn-193189": "BMP receptor ALK2/ALK3 inhibitor",
    "wye-125132": "mTORC1/2 inhibitor",
    "as-605240": "PI3K-gamma inhibitor",
    "tg-101348": "JAK2 inhibitor (fedratinib)",
    "fedratinib": "JAK2 inhibitor",
    "ruxolitinib": "JAK1/2 inhibitor",
    "tofacitinib": "JAK1/3 inhibitor",
    "baricitinib": "JAK1/2 inhibitor",
    "radicicol": "HSP90 inhibitor",
    "geldanamycin": "HSP90 inhibitor",
    "withaferin-a": "NF-kB/HSP90 inhibitor",
    "celastrol": "NF-kB/HSP90 inhibitor",
    "chelerythrine": "PKC inhibitor",
    "chelerythrine chloride": "PKC inhibitor",
    "lapatinib": "EGFR/HER2 dual inhibitor",
    "canertinib": "Pan-EGFR inhibitor",
    "gefitinib": "EGFR inhibitor",
    "afatinib": "Pan-EGFR inhibitor",
    "erlotinib": "EGFR inhibitor",
    "cetuximab": "EGFR monoclonal antibody",
    "cgp-60474": "CDK1/2 inhibitor",
    "palbociclib": "CDK4/6 inhibitor",
    "ribociclib": "CDK4/6 inhibitor",
    "abemaciclib": "CDK4/6 inhibitor",
    "bi-2536": "PLK1 inhibitor",
    "xmd-1150": "ERK5 inhibitor",
    "wz-3105": "SRC/ABL inhibitor",
    "wz-4-145": "CDK8 inhibitor",
    "saracatinib": "SRC/ALK2 dual inhibitor",
    "alvocidib": "CDK1/2/4/6/9 inhibitor (flavopiridol)",
    "at-7519": "CDK1/2/4/6/9 inhibitor",
    "pf-431396": "FAK/PYK2 inhibitor",
    "azd-5438": "CDK inhibitor",
    "azd-7762": "CHK1/2 inhibitor",
    "ql-xii-47": "MELK/FLT3 inhibitor",
    "apremilast": "PDE4 inhibitor",
    "roflumilast": "PDE4 inhibitor",
    "secukinumab": "IL-17A monoclonal antibody",
    "ixekizumab": "IL-17A monoclonal antibody",
    "guselkumab": "IL-23 monoclonal antibody",
    "risankizumab": "IL-23 monoclonal antibody",
    "ustekinumab": "IL-12/23 monoclonal antibody",
    "methotrexate": "Antifolate / immunosuppressant",
    "cyclosporine": "Calcineurin inhibitor",
    "acitretin": "Retinoid",
    "coal-tar": "Antiproliferative / keratolytic",
    "calcipotriol": "Vitamin D3 analogue",
    "betamethasone": "Corticosteroid",
    "deucravacitinib": "TYK2 inhibitor",
    "bimekizumab": "IL-17A/F dual inhibitor",
    "sotyktu": "TYK2 inhibitor",
    "mitoxantrone": "Topoisomerase II inhibitor",
    "plx-4720": "BRAF V600E inhibitor",
    "i-bet151": "BET bromodomain inhibitor",
    "i-bet": "BET bromodomain inhibitor",
    "linifanib": "VEGFR/PDGFR inhibitor",
}

def pubmed_hit_count(query, retries=3):
    params = {"db": "pubmed", "term": query, "rettype": "count",
              "retmode": "json", "email": NCBI_EMAIL}
    for attempt in range(retries):
        try:
            resp = requests.get(NCBI_ESEARCH, params=params, timeout=10)
            resp.raise_for_status()
            count = int(resp.json()["esearchresult"]["count"])
            time.sleep(NCBI_DELAY)
            return count
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(NCBI_DELAY * 3)
            else:
                return -1

def assess_novelty(compound_name):
    q = '"' + compound_name + '"'
    hits_pso  = pubmed_hit_count(q + ' AND (psoriasis OR "plaque psoriasis")')
    hits_kc   = pubmed_hit_count(q + ' AND ("keratinocyte proliferation" OR "keratinocyte hyperproliferation" OR psoriatic)')
    hits_il17 = pubmed_hit_count(q + ' AND ("IL-17" OR "IL-23" OR "TNF" OR "inflammatory skin")')
    if hits_pso == 0 and hits_kc == 0 and hits_il17 == 0:
        tier = "NOVEL_ALL"
    elif hits_pso == 0 and hits_kc == 0:
        tier = "NOVEL_PSORIASIS"
    else:
        tier = "KNOWN"
    return {"compound": compound_name,
            "hits_psoriasis": hits_pso,
            "hits_keratinocyte": hits_kc,
            "hits_il17_inflammation": hits_il17,
            "novelty_tier": tier}

def lookup_moa(name):
    return MOA_REFERENCE.get(name.lower().strip(), "unknown")

def main():
    print("=" * 60)
    print("PSORIASIS scRNA-seq PIPELINE")
    print("Script 07: Novelty & Priority Scoring")
    print("=" * 60)
    cand_path = os.path.join(PROCESSED_DIR, "lincs_candidates.csv")
    if not os.path.exists(cand_path):
        print("ERROR:", cand_path, "not found. Run 06_lincs_repurposing.py first.")
        sys.exit(1)
    print("\n[1/4] Loading LINCS candidates...")
    candidates = pd.read_csv(cand_path)
    print(" ", len(candidates), "candidates to assess")
    print("\n[2/4] Assessing novelty via PubMed...")
    print("  Querying", len(candidates), "compounds x 3 contexts...")
    novelty_rows = []
    for i, row in candidates.iterrows():
        compound = row["compound"]
        print("  [" + str(i+1) + "/" + str(len(candidates)) + "] " + compound + "...", end=" ", flush=True)
        nov = assess_novelty(compound)
        novelty_rows.append(nov)
        print(nov["novelty_tier"],
              "(Pso:" + str(nov["hits_psoriasis"]) +
              ", KC:" + str(nov["hits_keratinocyte"]) +
              ", IL17:" + str(nov["hits_il17_inflammation"]) + ")")
    novelty_df = pd.DataFrame(novelty_rows)
    novelty_df.to_csv(os.path.join(PROCESSED_DIR, "novelty_raw.csv"), index=False)
    print("\n[3/4] Computing priority scores...")
    merged = candidates.merge(novelty_df, on="compound", how="left")
    merged["novelty_tier"] = merged["novelty_tier"].fillna("KNOWN")
    merged["moa"] = merged["compound"].apply(lookup_moa)
    merged["priority_score"] = merged.apply(
        lambda r: round(r["max_reversal_score"] *
                        NOVELTY_WEIGHTS.get(r["novelty_tier"], 1.0) *
                        r["n_queries"], 1), axis=1)
    merged = merged.sort_values("priority_score", ascending=False)
    tier_counts = merged["novelty_tier"].value_counts()
    print("\n  Novelty breakdown:")
    for tier, count in tier_counts.items():
        print("    " + tier + ":", count, "compounds")
    display_cols = ["compound", "moa", "novelty_tier", "max_reversal_score", "n_queries", "priority_score"]
    print("\n  Top 20 priority candidates:")
    print(merged[display_cols].head(20).round(2).to_string(index=False))
    patent_watch = merged[
        (merged["novelty_tier"] == "NOVEL_ALL") &
        (merged["max_reversal_score"] >= PATENT_WATCH_MIN_REVERSAL) &
        (merged["n_queries"] >= PATENT_WATCH_MIN_QUERIES)].copy()
    print("\n  Patent watch list:", len(patent_watch), "compounds")
    if not patent_watch.empty:
        print(patent_watch[display_cols].to_string(index=False))
    print("\n[4/4] Saving...")
    merged.to_csv(os.path.join(PROCESSED_DIR, "priority_candidates.csv"), index=False)
    patent_watch.to_csv(os.path.join(PROCESSED_DIR, "patent_watch.csv"), index=False)
    print("\n" + "=" * 60)
    print("Script 07 complete.")
    print("  Priority candidates:", len(merged))
    print("  Patent watch:", len(patent_watch))
    print("=" * 60)
    print("\nPIPELINE COMPLETE. Review priority_candidates.csv for drug candidates.")

if __name__ == "__main__":
    main()
