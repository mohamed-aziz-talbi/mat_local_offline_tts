#!/usr/bin/env python3
"""
etl_and_warehouse.py
Senior Data Engineer ETL pipeline for BI project:
"The Economics of AI: Cloud API Vendor Lock-In vs. Local On-Premise TCO"

Reads  : data/raw/*.csv
Writes : data/processed/dim_provider.csv
         data/processed/dim_model_pricing.csv
         data/processed/dim_hardware.csv
         data/processed/fact_inference_cost.csv
         data/processed/bi_data_warehouse.db   (SQLite star schema)
"""

import csv
import os
import re
import sqlite3
import sys
import traceback

RAW_DIR = os.path.join(".", "data", "raw")
OUT_DIR = os.path.join(".", "data", "processed")
DB_PATH = os.path.join(OUT_DIR, "bi_data_warehouse.db")

# --- Business constants (from lab spec) ---
ELECTRICITY_RATE_USD_KWH = 0.15
ACTIVE_HOURS_PER_DAY = 8
DAYS_PER_MONTH = 30
GPU_UTILIZATION = 0.70           # TDP derating for realistic load
TCO_HORIZON_MONTHS = 36
GPT4O_BLENDED_USD_PER_1M = 2.50  # cloud baseline benchmark
AGENTIC_INFLATION_FACTOR = 15    # agents issue ~15x more inference calls
MONTHLY_TOKEN_VOLUME_SCENARIOS_M = [10, 50, 100, 500, 1000]

# Provider tags that serve open-weight models
OPEN_PROVIDER_TAGS = {
    "meta-llama", "mistralai", "deepseek", "qwen", "microsoft", "nvidia",
    "01-ai", "togethercomputer", "huggingfaceh4", "nousresearch", "tngtech",
    "unity", "liquid", "databricks", "eleutherai", "bigscience", "ibm",
    "ibm-granite", "openaccess-ai-collective", "cognitivecomputations",
    "sao10k", "undiess", "teknium", "laion", "stabilityai", "coreview",
}
# Model-id keywords that indicate open weights (catches google/gemma vs google/gemini)
OPEN_WEIGHT_KEYWORDS = (
    "llama", "mistral", "mixtral", "qwen", "gemma", "phi-", "phidown",
    "deepseek", "yi-", "redpajama", "olmo", "granite", "falcon", "mpt",
    "stablelm", "openchat", "hermes", "zephyr", "vicuna", "openllama",
    "tinyllama", "starcoder", "codellama", "wizardlm", "openelm", "smollm",
    "arctic", "jamba", "glm", "internlm", "command-r", "dbrx", "olmo",
    "开放", "bonsai",
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directory ensured: {path}")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows -> {path}")


def to_float(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def norm_key(s):
    """Normalize a model name/id for fuzzy joining."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_params_from_id(model_id):
    """Extract parameter count (B) from tokens like '8b', '70b', '1.5-7b', '405b'."""
    candidates = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", model_id.lower())
    if not candidates:
        return None
    vals = [float(c) for c in candidates]
    return max(vals)  # largest usually is the base model size


def classify_license(model_id, provider_tags):
    """Flag Open-Source vs Proprietary based on provider tags + model-id keywords."""
    mid = model_id.lower()
    tags = [t.strip().lower() for t in str(provider_tags).split(";") if t.strip()]
    if any(t in OPEN_PROVIDER_TAGS for t in tags):
        return "Open-Source"
    if any(k in mid for k in OPEN_WEIGHT_KEYWORDS):
        return "Open-Source"
    return "Proprietary"


# ---------------------------------------------------------------------------
# 1) Provider & Pricing dimensions
# ---------------------------------------------------------------------------
def build_pricing_dims(raw_models):
    print("\n=== Transform 1: Provider & Pricing Dimensions ===")
    dim_provider_rows = []
    dim_model_rows = []
    provider_stats = {}

    for i, m in enumerate(raw_models, start=1):
        model_id = m.get("id", "")
        name = m.get("name", "")
        prompt_1m = to_float(m.get("cost_per_1m_prompt_usd"))
        completion_1m = to_float(m.get("cost_per_1m_completion_usd"))
        provider_tags = m.get("provider_tags", "")
        provider = provider_tags.split(";")[0] if provider_tags else model_id.split("/")[0]
        context_length = to_float(m.get("context_length"), default=None)

        blended = round(prompt_1m * 0.75 + completion_1m * 0.25, 6)
        license_type = classify_license(model_id, provider_tags)

        dim_model_rows.append(
            {
                "Model_Key": i,
                "Model_Id": model_id,
                "Model_Name": name,
                "Provider_Name": provider,
                "Prompt_Cost_1M_USD": prompt_1m,
                "Completion_Cost_1M_USD": completion_1m,
                "Blended_Cost_1M_USD": blended,
                "Context_Length": int(context_length) if context_length else "",
                "License_Type": license_type,
                "Norm_Key": norm_key(model_id.split("/", 1)[-1]),
            }
        )

        st = provider_stats.setdefault(
            provider,
            {
                "models": 0, "blended": [], "open": 0, "prop": 0,
            },
        )
        st["models"] += 1
        if blended > 0:
            st["blended"].append(blended)
        if license_type == "Open-Source":
            st["open"] += 1
        else:
            st["prop"] += 1

    for pi, (provider, st) in enumerate(sorted(provider_stats.items()), start=1):
        avg_blended = round(sum(st["blended"]) / len(st["blended"]), 6) if st["blended"] else 0.0
        dim_provider_rows.append(
            {
                "Provider_Key": pi,
                "Provider_Name": provider,
                "Model_Count": st["models"],
                "Open_Source_Model_Count": st["open"],
                "Proprietary_Model_Count": st["prop"],
                "Avg_Blended_Cost_1M_USD": avg_blended,
                "Is_Open_Source_Focused": "Yes" if st["open"] >= st["prop"] else "No",
            }
        )

    write_csv(
        dim_provider_rows,
        os.path.join(OUT_DIR, "dim_provider.csv"),
        ["Provider_Key", "Provider_Name", "Model_Count", "Open_Source_Model_Count",
         "Proprietary_Model_Count", "Avg_Blended_Cost_1M_USD", "Is_Open_Source_Focused"],
    )
    write_csv(
        dim_model_rows,
        os.path.join(OUT_DIR, "dim_model_pricing.csv"),
        ["Model_Key", "Model_Id", "Model_Name", "Provider_Name", "Prompt_Cost_1M_USD",
         "Completion_Cost_1M_USD", "Blended_Cost_1M_USD", "Context_Length",
         "License_Type", "Norm_Key"],
    )
    return dim_provider_rows, dim_model_rows


# ---------------------------------------------------------------------------
# 2) Hardware dimension with TCO math
# ---------------------------------------------------------------------------
def build_hardware_dim(raw_gpus):
    print("\n=== Transform 2: Hardware Dimension (TCO) ===")
    rows = []
    for i, g in enumerate(raw_gpus, start=1):
        tdp = to_float(g.get("TDP_Watts"))
        purchase = to_float(g.get("Purchase_Price_USD"))
        monthly_power = round((tdp * GPU_UTILIZATION / 1000.0) * ACTIVE_HOURS_PER_DAY * DAYS_PER_MONTH * ELECTRICITY_RATE_USD_KWH, 4)
        tco_36 = round(purchase + monthly_power * TCO_HORIZON_MONTHS, 2)
        breakeven_tokens_1m = round(tco_36 / GPT4O_BLENDED_USD_PER_1M, 2)
        rows.append(
            {
                "Hardware_Key": i,
                "GPU_Name": g.get("GPU_Name", ""),
                "VRAM_GB": to_float(g.get("VRAM_GB")),
                "Purchase_Price_USD": purchase,
                "TDP_Watts": tdp,
                "Platform_Backend": g.get("Platform_Backend", ""),
                "Max_Quant_Param_Size_B": to_float(g.get("Max_Quant_Param_Size_B")),
                "Monthly_Power_USD": monthly_power,
                "TCO_36_Months_USD": tco_36,
                "Breakeven_Tokens_1M": breakeven_tokens_1m,
            }
        )
    write_csv(
        rows,
        os.path.join(OUT_DIR, "dim_hardware.csv"),
        ["Hardware_Key", "GPU_Name", "VRAM_GB", "Purchase_Price_USD", "TDP_Watts",
         "Platform_Backend", "Max_Quant_Param_Size_B", "Monthly_Power_USD",
         "TCO_36_Months_USD", "Breakeven_Tokens_1M"],
    )
    return rows


# ---------------------------------------------------------------------------
# 3) Fact table: models x fitting GPUs x monthly volume scenarios
# ---------------------------------------------------------------------------
def build_fact_table(dim_models, dim_hardware, dim_providers, hf_benchmarks):
    print("\n=== Transform 3: Fact_Inference_Cost (Star Schema Fact) ===")

    # HF parameter lookups:
    #   1) exact slug key (nk of the part after "/")
    #   2) suffix scan over all normalized HF names (OpenRouter slugs drop org
    #      words, e.g. "llama-3.3-70b" vs HF "Meta-Llama-3.3-70B")
    hf_by_slug = {}
    hf_pairs = []  # (norm_key, params) sorted longest-first for reliable suffix match
    for r in hf_benchmarks:
        p = to_float(r.get("parameter_count_b"), default=None)
        if not p:
            continue
        nm = str(r.get("model_name", ""))
        slug = nm.split("/", 1)[-1]
        hf_by_slug.setdefault(norm_key(slug), p)
        hf_pairs.append((norm_key(slug), p))
        hf_pairs.append((norm_key(nm), p))
    hf_pairs.sort(key=lambda x: -len(x[0]))
    print(f"  HF parameter lookups available: {len(hf_by_slug)}")

    def resolve_params(model_id, model_name=""):
        slug = model_id.split("/", 1)[-1]
        key = norm_key(slug)
        if key in hf_by_slug:
            return hf_by_slug[key]
        parsed = parse_params_from_id(model_id)
        if parsed:
            return parsed
        parsed = parse_params_from_id(model_name)
        if parsed:
            return parsed
        # Guarded suffix match: only for size-bearing slugs (e.g. "llama3370binstruct")
        # to avoid false positives on short generic names (e.g. "o1").
        if len(key) >= 5 and any(c.isdigit() for c in key):
            for fk, fp in hf_pairs:
                if len(fk) >= 6 and (fk.endswith(key) or key.endswith(fk)):
                    return fp
        return None

    provider_key_by_name = {p["Provider_Name"]: p["Provider_Key"] for p in dim_providers}

    fact_rows = []
    fid = 0
    models_with_params = 0

    for m in dim_models:
        blended = to_float(m["Blended_Cost_1M_USD"])
        if blended <= 0:
            continue  # exclude free-tier models from cost economics

        # Only open-weight models can be self-hosted => valid local-TCO comparison
        if m["License_Type"] != "Open-Source":
            continue

        params = resolve_params(m["Model_Id"], m["Model_Name"])
        if params is None:
            continue  # open weights but size unknown - cannot size local hardware
        models_with_params += 1

        fitting = [h for h in dim_hardware if to_float(h["Max_Quant_Param_Size_B"]) >= params]
        if not fitting:
            # smallest GPU still cannot host; use the largest as reference (offload/swap)
            fitting = [max(dim_hardware, key=lambda h: to_float(h["Max_Quant_Param_Size_B"]))]

        pkey = provider_key_by_name.get(m["Provider_Name"], 0)

        for h in fitting:
            monthly_capex = round(to_float(h["Purchase_Price_USD"]) / TCO_HORIZON_MONTHS, 4)
            local_opex = round(monthly_capex + to_float(h["Monthly_Power_USD"]), 4)
            for vol in MONTHLY_TOKEN_VOLUME_SCENARIOS_M:
                cloud_cost = round(blended * vol, 4)
                savings = round(cloud_cost - local_opex, 4)
                fid += 1
                fact_rows.append(
                    {
                        "Fact_Key": fid,
                        "Model_Key": m["Model_Key"],
                        "Model_Id": m["Model_Id"],
                        "Provider_Key": pkey,
                        "Hardware_Key": h["Hardware_Key"],
                        "GPU_Name": h["GPU_Name"],
                        "License_Type": m["License_Type"],
                        "Model_Params_B": params,
                        "Monthly_Token_Volume_M": vol,
                        "Blended_Cost_1M_USD": blended,
                        "Cloud_API_Monthly_Cost_USD": cloud_cost,
                        "Local_Hardware_Monthly_OpEx_USD": local_opex,
                        "Net_Monthly_Savings_USD": savings,
                        "Agentic_Inflated_Cost_15x_USD": round(cloud_cost * AGENTIC_INFLATION_FACTOR, 4),
                    }
                )

    print(f"  Models priced with known size: {models_with_params}")
    write_csv(
        fact_rows,
        os.path.join(OUT_DIR, "fact_inference_cost.csv"),
        ["Fact_Key", "Model_Key", "Model_Id", "Provider_Key", "Hardware_Key", "GPU_Name",
         "License_Type", "Model_Params_B", "Monthly_Token_Volume_M", "Blended_Cost_1M_USD",
         "Cloud_API_Monthly_Cost_USD", "Local_Hardware_Monthly_OpEx_USD",
         "Net_Monthly_Savings_USD", "Agentic_Inflated_Cost_15x_USD"],
    )
    return fact_rows


# ---------------------------------------------------------------------------
# 4) Load star schema into SQLite
# ---------------------------------------------------------------------------
def load_sqlite(dim_providers, dim_models, dim_hardware, fact_rows):
    print("\n=== Load 4: SQLite Data Warehouse ===")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"  Removed existing {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE Dim_Provider (
            Provider_Key INTEGER PRIMARY KEY,
            Provider_Name TEXT,
            Model_Count INTEGER,
            Open_Source_Model_Count INTEGER,
            Proprietary_Model_Count INTEGER,
            Avg_Blended_Cost_1M_USD REAL,
            Is_Open_Source_Focused TEXT
        )""")

    cur.execute("""
        CREATE TABLE Dim_Model (
            Model_Key INTEGER PRIMARY KEY,
            Model_Id TEXT,
            Model_Name TEXT,
            Provider_Name TEXT,
            Prompt_Cost_1M_USD REAL,
            Completion_Cost_1M_USD REAL,
            Blended_Cost_1M_USD REAL,
            Context_Length INTEGER,
            License_Type TEXT
        )""")

    cur.execute("""
        CREATE TABLE Dim_Hardware (
            Hardware_Key INTEGER PRIMARY KEY,
            GPU_Name TEXT,
            VRAM_GB REAL,
            Purchase_Price_USD REAL,
            TDP_Watts REAL,
            Platform_Backend TEXT,
            Max_Quant_Param_Size_B REAL,
            Monthly_Power_USD REAL,
            TCO_36_Months_USD REAL,
            Breakeven_Tokens_1M REAL
        )""")

    cur.execute("""
        CREATE TABLE Fact_Inference_Cost (
            Fact_Key INTEGER PRIMARY KEY,
            Model_Key INTEGER,
            Provider_Key INTEGER,
            Hardware_Key INTEGER,
            Model_Id TEXT,
            GPU_Name TEXT,
            License_Type TEXT,
            Model_Params_B REAL,
            Monthly_Token_Volume_M REAL,
            Blended_Cost_1M_USD REAL,
            Cloud_API_Monthly_Cost_USD REAL,
            Local_Hardware_Monthly_OpEx_USD REAL,
            Net_Monthly_Savings_USD REAL,
            Agentic_Inflated_Cost_15x_USD REAL,
            FOREIGN KEY (Model_Key) REFERENCES Dim_Model(Model_Key),
            FOREIGN KEY (Provider_Key) REFERENCES Dim_Provider(Provider_Key),
            FOREIGN KEY (Hardware_Key) REFERENCES Dim_Hardware(Hardware_Key)
        )""")

    cur.executemany(
        "INSERT INTO Dim_Provider VALUES (?,?,?,?,?,?,?)",
        [(p["Provider_Key"], p["Provider_Name"], p["Model_Count"], p["Open_Source_Model_Count"],
          p["Proprietary_Model_Count"], p["Avg_Blended_Cost_1M_USD"], p["Is_Open_Source_Focused"])
         for p in dim_providers],
    )
    cur.executemany(
        "INSERT INTO Dim_Model VALUES (?,?,?,?,?,?,?,?,?)",
        [(m["Model_Key"], m["Model_Id"], m["Model_Name"], m["Provider_Name"],
          m["Prompt_Cost_1M_USD"], m["Completion_Cost_1M_USD"], m["Blended_Cost_1M_USD"],
          int(m["Context_Length"]) if m["Context_Length"] else None, m["License_Type"])
         for m in dim_models],
    )
    cur.executemany(
        "INSERT INTO Dim_Hardware VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(h["Hardware_Key"], h["GPU_Name"], h["VRAM_GB"], h["Purchase_Price_USD"],
          h["TDP_Watts"], h["Platform_Backend"], h["Max_Quant_Param_Size_B"],
          h["Monthly_Power_USD"], h["TCO_36_Months_USD"], h["Breakeven_Tokens_1M"])
         for h in dim_hardware],
    )
    cur.executemany(
        "INSERT INTO Fact_Inference_Cost VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(f["Fact_Key"], f["Model_Key"], f["Provider_Key"], f["Hardware_Key"], f["Model_Id"],
          f["GPU_Name"], f["License_Type"], f["Model_Params_B"], f["Monthly_Token_Volume_M"],
          f["Blended_Cost_1M_USD"], f["Cloud_API_Monthly_Cost_USD"],
          f["Local_Hardware_Monthly_OpEx_USD"], f["Net_Monthly_Savings_USD"],
          f["Agentic_Inflated_Cost_15x_USD"])
         for f in fact_rows],
    )

    conn.commit()

    for t in ["Dim_Provider", "Dim_Model", "Dim_Hardware", "Fact_Inference_Cost"]:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  Table {t}: {cur.fetchone()[0]} rows")
    conn.close()
    print(f"  SQLite warehouse saved -> {os.path.abspath(DB_PATH)}")


# ---------------------------------------------------------------------------
def main():
    print("ETL & Warehouse Build")
    print("Project: The Economics of AI: Cloud API Vendor Lock-In vs. Local On-Premise TCO")
    print("=" * 78)
    ensure_dir(OUT_DIR)

    raw_models = read_csv(os.path.join(RAW_DIR, "raw_openrouter_pricing.csv"))
    raw_gpus = read_csv(os.path.join(RAW_DIR, "raw_gpu_hardware.csv"))
    raw_hf = read_csv(os.path.join(RAW_DIR, "raw_huggingface_benchmarks.csv"))
    print(f"Loaded raw: {len(raw_models)} models, {len(raw_gpus)} GPUs, {len(raw_hf)} HF benchmarks")

    dim_providers, dim_models = build_pricing_dims(raw_models)
    dim_hardware = build_hardware_dim(raw_gpus)
    fact_rows = build_fact_table(dim_models, dim_hardware, dim_providers, raw_hf)
    load_sqlite(dim_providers, dim_models, dim_hardware, fact_rows)

    print("\n" + "=" * 78)
    print("ETL COMPLETE")
    print(f"  Processed directory: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
