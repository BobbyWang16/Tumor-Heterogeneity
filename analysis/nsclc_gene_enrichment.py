from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from scipy.stats import ttest_ind


BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "NSCLC-Radiogenomics"
GENE_FILE = DATA_DIR / "gene" / "GSE103584_R01_NSCLC_RNAseq.txt"
CLINICAL_FILE = DATA_DIR / "clinical.xlsx"
OUT_DIR = BASE / "analysis" / "nsclc_gene_enrichment_results"
OUT_XLSX = OUT_DIR / "nsclc_gene_pathway_enrichment.xlsx"

MAX_RAW_VALUE = 1_000_000
MIN_DETECTION_RATE = 0.20
STRICT_FDR = 0.05
STRICT_ABS_LOG2FC = 1.0
MIN_ENRICH_GENES = 10
FALLBACK_TOP_N = 200
ENRICHR_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
]


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    out = np.ones_like(p, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(vals)
    tmp[order] = q
    out[ok] = tmp
    return out


def clean_text_series(series):
    return series.astype(str).str.strip().replace({"nan": np.nan, "NaN": np.nan})


def safe_sheet_name(name):
    bad = '[]:*?/\\'
    for ch in bad:
        name = name.replace(ch, "_")
    return name[:31]


def load_inputs():
    expr = pd.read_csv(GENE_FILE, sep="\t", index_col=0, na_values=["NA"])
    expr.index.name = "gene"
    clinical = pd.read_excel(CLINICAL_FILE)
    clinical.columns = [str(c).strip() for c in clinical.columns]
    clinical["Case ID"] = clean_text_series(clinical["Case ID"])
    clinical = clinical.drop_duplicates("Case ID").set_index("Case ID")
    common = expr.columns[expr.columns.isin(clinical.index)]
    if common.empty:
        raise ValueError("No shared case IDs between expression and clinical data")
    return expr.loc[:, common], clinical.loc[common].copy()


def preprocess_expression(expr):
    raw_max = expr.max(axis=1, skipna=True)
    detection_rate = expr.notna().mean(axis=1)
    keep = raw_max.notna() & (raw_max <= MAX_RAW_VALUE) & (detection_rate >= MIN_DETECTION_RATE)
    filtered = expr.loc[keep].copy()
    log_expr = np.log2(filtered.fillna(0).clip(lower=0) + 1)

    qc = pd.DataFrame(
        {
            "metric": [
                "raw_genes",
                "raw_samples",
                "raw_missing_values",
                "raw_missing_rate",
                "genes_removed_all_or_low_detection",
                "genes_removed_extreme_raw_value",
                "genes_kept_for_testing",
                "samples_matched_to_clinical",
                "missing_values_handling",
                "transform",
            ],
            "value": [
                expr.shape[0],
                expr.shape[1],
                int(expr.isna().sum().sum()),
                float(expr.isna().mean().mean()),
                int(((detection_rate < MIN_DETECTION_RATE) | raw_max.isna()).sum()),
                int((raw_max > MAX_RAW_VALUE).sum()),
                int(keep.sum()),
                expr.shape[1],
                "NA replaced with 0 after gene filtering",
                "log2(expression + 1)",
            ],
        }
    )
    extreme = (
        pd.DataFrame(
            {
                "gene": expr.index,
                "raw_max": raw_max.values,
                "detection_rate": detection_rate.values,
            }
        )
        .sort_values("raw_max", ascending=False)
        .head(300)
    )
    return filtered, log_expr, qc, extreme


def clinical_counts(matched):
    fields = [
        "Histology",
        "Histopathological Grade",
        "Gender",
        "EGFR mutation status",
        "KRAS mutation status",
        "ALK translocation status",
        "Recurrence",
        "Survival Status",
        "event",
    ]
    rows = []
    for field in fields:
        if field not in matched.columns:
            continue
        counts = matched[field].fillna("NA").astype(str).str.strip().value_counts(dropna=False)
        for level, n in counts.items():
            rows.append({"clinical_field": field, "level": level, "n": int(n)})
    return pd.DataFrame(rows)


def make_contrasts(matched):
    return [
        {
            "name": "Histology_Adeno_vs_Squamous",
            "clinical_field": "Histology",
            "case_label": "Adenocarcinoma",
            "control_label": "Squamous cell carcinoma",
        },
        {
            "name": "EGFR_Mutant_vs_Wildtype",
            "clinical_field": "EGFR mutation status",
            "case_label": "Mutant",
            "control_label": "Wildtype",
        },
        {
            "name": "KRAS_Mutant_vs_Wildtype",
            "clinical_field": "KRAS mutation status",
            "case_label": "Mutant",
            "control_label": "Wildtype",
        },
        {
            "name": "Recurrence_Yes_vs_No",
            "clinical_field": "Recurrence",
            "case_label": "yes",
            "control_label": "no",
        },
        {
            "name": "Death_Event_vs_NoEvent",
            "clinical_field": "event",
            "case_label": 1,
            "control_label": 0,
        },
    ]


def differential_for_contrast(raw_filtered, log_expr, matched, contrast):
    field = contrast["clinical_field"]
    case = contrast["case_label"]
    control = contrast["control_label"]
    labels = matched[field]

    if field == "event":
        case_mask = labels == case
        control_mask = labels == control
    else:
        label_text = clean_text_series(labels)
        case_mask = label_text == str(case)
        control_mask = label_text == str(control)

    case_samples = labels.index[case_mask].tolist()
    control_samples = labels.index[control_mask].tolist()
    x_case = log_expr[case_samples]
    x_ctrl = log_expr[control_samples]

    t_stat, pvalue = ttest_ind(
        x_case.to_numpy(),
        x_ctrl.to_numpy(),
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )
    pvalue = np.where(np.isfinite(pvalue), pvalue, 1.0)
    t_stat = np.where(np.isfinite(t_stat), t_stat, 0.0)

    mean_case = x_case.mean(axis=1)
    mean_ctrl = x_ctrl.mean(axis=1)
    log2fc = mean_case - mean_ctrl
    qvalue = bh_fdr(pvalue)
    det_case = raw_filtered[case_samples].notna().mean(axis=1)
    det_ctrl = raw_filtered[control_samples].notna().mean(axis=1)

    result = pd.DataFrame(
        {
            "contrast": contrast["name"],
            "gene": log_expr.index,
            "clinical_field": field,
            "case_label": str(case),
            "control_label": str(control),
            "n_case": len(case_samples),
            "n_control": len(control_samples),
            "mean_log2_case": mean_case.values,
            "mean_log2_control": mean_ctrl.values,
            "log2FC_case_vs_control": log2fc.values,
            "t_statistic": t_stat,
            "p_value": pvalue,
            "fdr_bh": qvalue,
            "detection_rate_case": det_case.values,
            "detection_rate_control": det_ctrl.values,
        }
    ).sort_values(["fdr_bh", "p_value", "gene"])
    result["direction"] = np.where(result["log2FC_case_vs_control"] >= 0, "UP", "DOWN")
    result["strict_deg"] = (result["fdr_bh"] < STRICT_FDR) & (
        result["log2FC_case_vs_control"].abs() >= STRICT_ABS_LOG2FC
    )
    return result


def choose_enrichment_genes(de, direction):
    signed = de[de["direction"] == direction].copy()
    strict = signed[signed["strict_deg"]].sort_values(["fdr_bh", "p_value"])
    if len(strict) >= MIN_ENRICH_GENES:
        selected = strict
        method = f"strict: FDR<{STRICT_FDR}, abs(log2FC)>={STRICT_ABS_LOG2FC}"
    else:
        selected = signed.sort_values(["p_value", "fdr_bh"]).head(FALLBACK_TOP_N)
        method = f"fallback: top {FALLBACK_TOP_N} directional genes by p-value"
    return selected["gene"].astype(str).tolist(), method, int(len(strict))


def enrichr_for_genes(session, genes, description):
    if len(genes) < 5:
        return pd.DataFrame(), f"Skipped: only {len(genes)} genes"

    add_url = "https://maayanlab.cloud/Enrichr/addList"
    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    payload = {"list": (None, "\n".join(genes)), "description": (None, description)}
    response = session.post(add_url, files=payload, timeout=30)
    if response.status_code != 200:
        return pd.DataFrame(), f"addList HTTP {response.status_code}: {response.text[:200]}"
    user_list_id = response.json()["userListId"]

    frames = []
    errors = []
    for library in ENRICHR_LIBRARIES:
        try:
            r = session.get(
                enrich_url,
                params={"userListId": user_list_id, "backgroundType": library},
                timeout=60,
            )
            if r.status_code != 200:
                errors.append(f"{library}: HTTP {r.status_code}")
                continue
            rows = r.json().get(library, [])
            parsed = []
            for row in rows:
                genes_overlap = row[5]
                if isinstance(genes_overlap, list):
                    genes_overlap = ";".join(genes_overlap)
                parsed.append(
                    {
                        "library": library,
                        "rank": row[0],
                        "term": row[1],
                        "p_value": row[2],
                        "z_score": row[3],
                        "combined_score": row[4],
                        "overlap_genes": genes_overlap,
                        "adjusted_p_value": row[6],
                    }
                )
            frames.append(pd.DataFrame(parsed))
            time.sleep(0.2)
        except Exception as exc:
            errors.append(f"{library}: {type(exc).__name__}: {exc}")

    if not frames:
        return pd.DataFrame(), "; ".join(errors) if errors else "No enrichment returned"
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["adjusted_p_value", "p_value", "library", "rank"]), "; ".join(errors)


def plot_volcano(de, path):
    df = de.copy()
    df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))
    colors = np.where(
        (df["fdr_bh"] < STRICT_FDR) & (df["log2FC_case_vs_control"] >= STRICT_ABS_LOG2FC),
        "#b2182b",
        np.where(
            (df["fdr_bh"] < STRICT_FDR) & (df["log2FC_case_vs_control"] <= -STRICT_ABS_LOG2FC),
            "#2166ac",
            "#9a9a9a",
        ),
    )
    plt.figure(figsize=(7.0, 5.2), dpi=160)
    plt.scatter(df["log2FC_case_vs_control"], df["neg_log10_p"], s=7, c=colors, alpha=0.65, linewidths=0)
    plt.axvline(STRICT_ABS_LOG2FC, color="#555555", linestyle="--", linewidth=0.8)
    plt.axvline(-STRICT_ABS_LOG2FC, color="#555555", linestyle="--", linewidth=0.8)
    plt.xlabel("log2FC case vs control")
    plt.ylabel("-log10(p-value)")
    plt.title(de["contrast"].iloc[0])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_enrichment_bar(enrich, path, title):
    sig = enrich[enrich["adjusted_p_value"] < 0.05].copy()
    if sig.empty:
        sig = enrich.sort_values("p_value").head(12).copy()
    else:
        sig = sig.sort_values("adjusted_p_value").head(12)
    if sig.empty:
        return
    sig["label"] = sig["library"].str.replace("_", " ", regex=False) + " | " + sig["term"].astype(str)
    sig["score"] = -np.log10(sig["adjusted_p_value"].clip(lower=1e-300))
    sig = sig.iloc[::-1]
    plt.figure(figsize=(8.5, max(4.0, 0.38 * len(sig))), dpi=160)
    plt.barh(sig["label"], sig["score"], color="#3a6ea5")
    plt.xlabel("-log10 adjusted p-value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def format_workbook(path):
    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
        for col_cells in ws.columns:
            letter = col_cells[0].column_letter
            max_len = 0
            for cell in col_cells[:200]:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 45)
    wb.save(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr, matched = load_inputs()
    raw_filtered, log_expr, qc, extreme = preprocess_expression(expr)
    counts = clinical_counts(matched)

    all_de = []
    summary_rows = []
    gene_list_rows = []
    enrichment_rows = []
    enrichment_errors = []

    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    for contrast in make_contrasts(matched):
        de = differential_for_contrast(raw_filtered, log_expr, matched, contrast)
        all_de.append(de)
        plot_volcano(de, figures_dir / f"volcano_{contrast['name']}.png")

        n_up = int(((de["fdr_bh"] < STRICT_FDR) & (de["log2FC_case_vs_control"] >= STRICT_ABS_LOG2FC)).sum())
        n_down = int(((de["fdr_bh"] < STRICT_FDR) & (de["log2FC_case_vs_control"] <= -STRICT_ABS_LOG2FC)).sum())
        summary_rows.append(
            {
                "contrast": contrast["name"],
                "clinical_field": contrast["clinical_field"],
                "case_label": str(contrast["case_label"]),
                "control_label": str(contrast["control_label"]),
                "n_case": int(de["n_case"].iloc[0]),
                "n_control": int(de["n_control"].iloc[0]),
                "strict_up_genes": n_up,
                "strict_down_genes": n_down,
                "strict_total_genes": n_up + n_down,
            }
        )

        for direction in ["UP", "DOWN"]:
            genes, method, strict_count = choose_enrichment_genes(de, direction)
            for rank, gene in enumerate(genes, start=1):
                gene_list_rows.append(
                    {
                        "contrast": contrast["name"],
                        "direction": direction,
                        "selection_method": method,
                        "strict_deg_count_for_direction": strict_count,
                        "selected_rank": rank,
                        "gene": gene,
                    }
                )

    de_all = pd.concat(all_de, ignore_index=True)
    strict_de = de_all[de_all["strict_deg"]].copy()
    summary = pd.DataFrame(summary_rows)
    gene_lists = pd.DataFrame(gene_list_rows)

    session = requests.Session()
    for (contrast_name, direction), genes_df in gene_lists.groupby(["contrast", "direction"], sort=False):
        genes = genes_df["gene"].tolist()
        description = f"{contrast_name}_{direction}"
        enrich, err = enrichr_for_genes(session, genes, description)
        if err:
            enrichment_errors.append(
                {
                    "contrast": contrast_name,
                    "direction": direction,
                    "message": err,
                }
            )
        if enrich.empty:
            continue
        enrich.insert(0, "contrast", contrast_name)
        enrich.insert(1, "direction", direction)
        enrich.insert(2, "input_gene_count", len(genes))
        enrichment_rows.append(enrich)
        plot_enrichment_bar(enrich, figures_dir / f"enrichment_{contrast_name}_{direction}.png", description)

    enrichment_all = pd.concat(enrichment_rows, ignore_index=True) if enrichment_rows else pd.DataFrame()
    enrichment_top = (
        enrichment_all[enrichment_all["adjusted_p_value"] < 0.05]
        .sort_values(["contrast", "direction", "adjusted_p_value", "p_value"])
        .copy()
        if not enrichment_all.empty
        else pd.DataFrame()
    )
    errors = pd.DataFrame(enrichment_errors)

    readme = pd.DataFrame(
        {
            "item": [
                "expression_file",
                "clinical_file_used",
                "sample_match",
                "contrasts",
                "preprocessing",
                "gene_filter",
                "differential_test",
                "strict_DEG_definition",
                "enrichment_method",
                "enrichment_libraries",
                "direction_definition",
            ],
            "value": [
                str(GENE_FILE.relative_to(BASE)),
                str(CLINICAL_FILE.relative_to(BASE)),
                "130 RNA-seq samples matched exactly to clinical Case ID",
                "; ".join(summary["contrast"].tolist()),
                "NA -> 0 after filtering; log2(expression + 1)",
                f"detection rate >= {MIN_DETECTION_RATE}; raw max <= {MAX_RAW_VALUE}",
                "Welch t-test on log2 expression",
                f"BH FDR < {STRICT_FDR} and abs(log2FC) >= {STRICT_ABS_LOG2FC}",
                "Enrichr over-representation analysis; strict DEG list if >=10 genes, otherwise top 200 directional genes by p-value",
                "; ".join(ENRICHR_LIBRARIES),
                "UP means higher in case_label than control_label; DOWN means higher in control_label",
            ],
        }
    )

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        qc.to_excel(writer, sheet_name="Expression_QC", index=False)
        counts.to_excel(writer, sheet_name="Clinical_Counts", index=False)
        matched.reset_index().rename(columns={"index": "Case ID"}).to_excel(writer, sheet_name="Matched_Clinical", index=False)
        extreme.to_excel(writer, sheet_name="Extreme_Genes", index=False)
        summary.to_excel(writer, sheet_name="DEG_Summary", index=False)
        strict_de.to_excel(writer, sheet_name="Strict_DEGs", index=False)
        de_all.to_excel(writer, sheet_name="All_DE_Results", index=False)
        gene_lists.to_excel(writer, sheet_name="Enrich_Input_Genes", index=False)
        enrichment_all.to_excel(writer, sheet_name="Enrichment_All", index=False)
        enrichment_top.to_excel(writer, sheet_name="Enrichment_FDR05", index=False)
        errors.to_excel(writer, sheet_name="Enrichr_Errors", index=False)

    format_workbook(OUT_XLSX)
    de_all.to_csv(OUT_DIR / "all_differential_results.csv", index=False)
    strict_de.to_csv(OUT_DIR / "strict_differential_genes.csv", index=False)
    enrichment_all.to_csv(OUT_DIR / "enrichment_all.csv", index=False)
    enrichment_top.to_csv(OUT_DIR / "enrichment_fdr05.csv", index=False)
    summary.to_csv(OUT_DIR / "deg_summary.csv", index=False)

    print(f"wrote {OUT_XLSX}")
    print(summary.to_string(index=False))
    if enrichment_top.empty:
        print("No enrichment terms passed adjusted p-value < 0.05")
    else:
        print("\nTop enriched terms:")
        cols = ["contrast", "direction", "library", "term", "adjusted_p_value", "overlap_genes"]
        print(enrichment_top[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
