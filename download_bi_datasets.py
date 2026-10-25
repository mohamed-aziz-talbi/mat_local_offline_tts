#!/usr/bin/env python3
"""
download_bi_datasets.py
BI Project: "The Economics of AI: Cloud API Vendor Lock-In vs. Local On-Premise TCO"

Extracts/generates three raw datasets into ./data/raw/:
  1. raw_openrouter_pricing.csv      - OpenRouter live API pricing
  2. raw_huggingface_benchmarks.csv  - Hugging Face LLM benchmark metrics
  3. raw_gpu_hardware.csv            - Consumer GPU / accelerator specs
"""

import csv
import json
import os
import sys
import traceback

# ---------------------------------------------------------------------------
# Optional imports handled gracefully
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    requests = None

try:
    from datasets import load_dataset  # Hugging Face datasets library
except ImportError:
    load_dataset = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow.parquet as _pq
    from io import BytesIO as _BytesIO
except ImportError:
    _pq = None


DATA_RAW_DIR = os.path.join(".", "data", "raw")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    print(f"Directory ensured: {path}")


def save_csv(data, filepath, fieldnames):
    """Write list-of-dicts to CSV, return True/False."""
    try:
        ensure_dir(os.path.dirname(filepath) or ".")
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print(f"Saved {len(data)} rows -> {filepath}")
        return True
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# 1) OpenRouter Live API Pricing
# ---------------------------------------------------------------------------
def extract_openrouter_pricing(output_dir=DATA_RAW_DIR) -> bool:
    print("\n=== Extracting OpenRouter Live API Pricing Data ===")
    url = "https://openrouter.ai/api/v1/models"
    data_list = []

    fieldnames = [
        "id",
        "name",
        "pricing.prompt",
        "pricing.completion",
        "cost_per_1m_prompt_usd",
        "cost_per_1m_completion_usd",
        "context_length",
        "provider_tags",
    ]

    try:
        if requests is None:
            raise ImportError(
                "The 'requests' library is not installed. "
                "Install it via: pip install requests"
            )

        headers = {
            "User-Agent": "BI-Data-Pipeline/1.0",
            "Accept": "application/json",
        }
        print(f"GET {url} ...")
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        body = resp.json()

        models = body.get("data", []) if isinstance(body, dict) else []
        print(f"Fetched {len(models)} models from OpenRouter.")

        for m in models:
            try:
                id_ = m.get("id") or ""
                name = m.get("name") or id_
                pricing = m.get("pricing") or {}
                prompt_per_token = pricing.get("prompt")
                completion_per_token = pricing.get("completion")
                context_length = m.get("context_length")

                # Convert token prices to "Cost per 1M Tokens" in USD
                cost_per_m_prompt = ""
                cost_per_m_completion = ""
                try:
                    if prompt_per_token is not None:
                        cost_per_m_prompt = round(float(prompt_per_token) * 1_000_000, 6)
                except (TypeError, ValueError):
                    cost_per_m_prompt = ""
                try:
                    if completion_per_token is not None:
                        cost_per_m_completion = round(
                            float(completion_per_token) * 1_000_000, 6
                        )
                except (TypeError, ValueError):
                    cost_per_m_completion = ""

                # Provider tags: derive vendor prefix from the model id
                # (e.g. "openai/gpt-4o" -> "openai", "deepseek/..." -> "deepseek")
                # and merge any explicit provider fields if the API provides them.
                provider_tags = []
                if "/" in id_:
                    provider_tags.append(id_.split("/", 1)[0])
                arch = m.get("architecture")
                if isinstance(arch, dict):
                    prov = arch.get("provider") or arch.get("providers")
                    if isinstance(prov, list):
                        provider_tags.extend(prov)
                    elif isinstance(prov, str):
                        provider_tags.append(prov)
                prov_top = m.get("providers")
                if isinstance(prov_top, list):
                    provider_tags.extend(prov_top)
                elif isinstance(prov_top, str):
                    provider_tags.append(prov_top)

                # De-duplicate while preserving order
                seen = set()
                unique_tags = []
                for p in provider_tags:
                    if p not in seen:
                        seen.add(p)
                        unique_tags.append(p)

                data_list.append(
                    {
                        "id": id_,
                        "name": name,
                        "pricing.prompt": prompt_per_token
                        if prompt_per_token is not None
                        else "",
                        "pricing.completion": completion_per_token
                        if completion_per_token is not None
                        else "",
                        "cost_per_1m_prompt_usd": cost_per_m_prompt,
                        "cost_per_1m_completion_usd": cost_per_m_completion,
                        "context_length": context_length
                        if context_length is not None
                        else "",
                        "provider_tags": ";".join(unique_tags),
                    }
                )
            except Exception as ex:
                print(f"  Skipping model due to error: {ex}")
                continue

        return save_csv(data_list, os.path.join(output_dir, "raw_openrouter_pricing.csv"), fieldnames)

    except Exception as e:
        print(f"Error extracting OpenRouter data: {e}")
        traceback.print_exc()
        if data_list:
            print("Attempting to save partial data...")
            return save_csv(
                data_list, os.path.join(output_dir, "raw_openrouter_pricing.csv"), fieldnames
            )
        return False


# ---------------------------------------------------------------------------
# 2) Hugging Face Benchmark Data
# ---------------------------------------------------------------------------
def _parse_leaderboard_row(row: dict) -> dict:
    """Normalize one open-llm-leaderboard/contents row into our schema."""
    model_name = (
        row.get("fullname")
        or row.get("Model")
        or row.get("model")
        or row.get("name")
        or ""
    )
    # Strip any leftover HTML if Model column was used
    if "<" in str(model_name):
        model_name = row.get("fullname", model_name)
    params = None
    for k in ["#Params (B)", "Parameters", "parameters", "Params (B)", "param_count", "size"]:
        if row.get(k) is not None:
            try:
                v = row[k]
                if isinstance(v, str):
                    v = v.replace("B", "").strip()
                params = float(v)
            except (TypeError, ValueError):
                params = None
            break
    # MMLU benchmark (new leaderboard publishes MMLU-PRO; fall back to classic MMLU keys)
    mmlu = None
    for k in ["MMLU-PRO", "MMLU", "mmlu", "MMLU (5-shot)", "MMLU Score", "acc_norm"]:
        if row.get(k) is not None:
            try:
                mmlu = float(row[k])
            except (TypeError, ValueError):
                mmlu = None
            break
    return {
        "model_name": model_name,
        "parameter_count_b": params if params is not None else "",
        "mmlu_score": mmlu if mmlu is not None else "",
    }


def extract_huggingface_benchmarks(output_dir=DATA_RAW_DIR) -> bool:
    print("\n=== Extracting Hugging Face Benchmark Data ===")
    data_list = []

    fieldnames = ["model_name", "parameter_count_b", "mmlu_score"]

    # --- Strategy A: download pre-built parquet from the HF Hub (fastest, full data) ---
    if _pq is not None:
        try:
            if requests is None:
                raise ImportError("requests not available")
            print("Trying HF Hub parquet export for open-llm-leaderboard/contents ...")
            meta = requests.get(
                "https://huggingface.co/api/datasets/open-llm-leaderboard/contents/parquet/default/train",
                headers={"User-Agent": "BI-Data-Pipeline/1.0"},
                timeout=60,
            )
            meta.raise_for_status()
            parquet_urls = meta.json()
            if isinstance(parquet_urls, str):
                parquet_urls = [parquet_urls]
            for url in parquet_urls:
                print(f"  Downloading {url} ...")
                dl = requests.get(url, headers={"User-Agent": "BI-Data-Pipeline/1.0"}, timeout=120)
                dl.raise_for_status()
                table = _pq.read_table(_BytesIO(dl.content))
                rows = table.to_pylist()
                print(f"  Read {len(rows)} rows from parquet")
                for row in rows:
                    try:
                        parsed = _parse_leaderboard_row(row)
                        if parsed["model_name"]:
                            data_list.append(parsed)
                    except Exception:
                        continue
            if data_list:
                return save_csv(
                    data_list,
                    os.path.join(output_dir, "raw_huggingface_benchmarks.csv"),
                    fieldnames,
                )
        except Exception as e:
            print(f"  Parquet download failed: {e}")
            traceback.print_exc()

    # --- Strategy B: use the Hugging Face 'datasets' library ---
    if load_dataset is not None:
        try:
            print("Loading HF dataset: open-llm-leaderboard/contents ...")
            ds = load_dataset("open-llm-leaderboard/contents", split="train")
            print(f"  Loaded open-llm-leaderboard/contents ({len(ds)} rows)")
            for row in ds:
                try:
                    parsed = _parse_leaderboard_row(row)
                    if parsed["model_name"]:
                        data_list.append(parsed)
                except Exception:
                    continue
            if data_list:
                return save_csv(
                    data_list,
                    os.path.join(output_dir, "raw_huggingface_benchmarks.csv"),
                    fieldnames,
                )
        except Exception as e:
            print(f"  Error using datasets library: {e}")
            traceback.print_exc()
    else:
        print("'datasets' library not installed; using datasets-server REST API...")

    # --- Strategy B: Hugging Face datasets-server REST API (paginated) ---
    try:
        if requests is None:
            raise ImportError("requests not available")

        dataset_id = "open-llm-leaderboard/contents"
        base = "https://datasets-server.huggingface.co/rows"
        headers = {"User-Agent": "BI-Data-Pipeline/1.0"}

        # Get total row count first
        params = {
            "dataset": dataset_id,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 1,
        }
        probe = requests.get(base, headers=headers, params=params, timeout=60)
        probe.raise_for_status()
        total = probe.json().get("num_rows_total", 0)
        print(f"datasets-server reports {total} rows for {dataset_id}")

        page_size = 100  # API max per request
        offset = 0
        max_retries = 5
        while offset < total:
            params = {
                "dataset": dataset_id,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": page_size,
            }
            resp = None
            payload = None
            for attempt in range(max_retries):
                try:
                    resp = requests.get(base, headers=headers, params=params, timeout=60)
                    resp.raise_for_status()
                    payload = resp.json()  # may raise if rate-limit HTML is returned
                    break
                except Exception as req_err:
                    status = getattr(req_err, "response", None)
                    code = status.status_code if status is not None else None
                    if code == 429 or (code is not None and code >= 500) or isinstance(
                        req_err, ValueError
                    ):
                        # 429 / 5xx / JSON decode error -> likely throttled
                        wait = 2 ** attempt  # exponential backoff: 1,2,4,8,16s
                        print(f"  Transient error at offset {offset} ({type(req_err).__name__}); retry {attempt + 1}/{max_retries} in {wait}s...")
                        import time as _time

                        _time.sleep(wait)
                        continue
                    raise
            if payload is None:
                print(f"  Giving up on offset {offset} after {max_retries} retries; saving partial data.")
                break
            rows = payload.get("rows", [])
            if not rows:
                break
            for item in rows:
                row = item.get("row", {}) if isinstance(item, dict) else {}
                try:
                    parsed = _parse_leaderboard_row(row)
                    if parsed["model_name"]:
                        data_list.append(parsed)
                except Exception:
                    continue
            offset += page_size
            print(f"  Fetched rows {min(offset, total)}/{total}...")
            import time as _time

            _time.sleep(0.5)  # be polite to the API to avoid 429s

        if data_list:
            return save_csv(
                data_list,
                os.path.join(output_dir, "raw_huggingface_benchmarks.csv"),
                fieldnames,
            )
    except Exception as e:
        print(f"Error in datasets-server REST API fallback: {e}")
        traceback.print_exc()

    # --- Save whatever partial data we collected, or a header-only template ---
    if data_list:
        print(f"Saving partial data ({len(data_list)} rows).")
        return save_csv(
            data_list,
            os.path.join(output_dir, "raw_huggingface_benchmarks.csv"),
            fieldnames,
        )
    print("Saving template-only file (no live data fetched).")
    return save_csv(
        [], os.path.join(output_dir, "raw_huggingface_benchmarks.csv"), fieldnames
    )


# ---------------------------------------------------------------------------
# 3) Consumer Hardware Specifications Dataset (static generation)
# ---------------------------------------------------------------------------
def generate_gpu_hardware(output_dir=DATA_RAW_DIR) -> bool:
    print("\n=== Generating GPU Hardware Specifications Dataset ===")
    data_list = [
        # NVIDIA GPUs (CUDA)
        {"GPU_Name": "RTX 2070 (8GB)", "VRAM_GB": 8, "Purchase_Price_USD": 499.0, "TDP_Watts": 175, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 8.0},
        {"GPU_Name": "RTX 2080 (8GB)", "VRAM_GB": 8, "Purchase_Price_USD": 699.0, "TDP_Watts": 215, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 8.0},
        {"GPU_Name": "RTX 3060 (12GB)", "VRAM_GB": 12, "Purchase_Price_USD": 329.0, "TDP_Watts": 170, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 12.0},
        {"GPU_Name": "RTX 3070 (8GB)", "VRAM_GB": 8, "Purchase_Price_USD": 499.0, "TDP_Watts": 220, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 8.0},
        {"GPU_Name": "RTX 4060 (8GB)", "VRAM_GB": 8, "Purchase_Price_USD": 299.0, "TDP_Watts": 115, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 8.0},
        # AMD GPUs (ROCm)
        {"GPU_Name": "RX 6600 (8GB)", "VRAM_GB": 8, "Purchase_Price_USD": 269.0, "TDP_Watts": 132, "Platform_Backend": "ROCm", "Max_Quant_Param_Size_B": 8.0},
        {"GPU_Name": "RX 6700 XT (12GB)", "VRAM_GB": 12, "Purchase_Price_USD": 479.0, "TDP_Watts": 230, "Platform_Backend": "ROCm", "Max_Quant_Param_Size_B": 12.0},
        # Intel GPUs (SYCL / oneAPI)
        {"GPU_Name": "Arc A750 (8GB)", "VRAM_GB": 8, "Purchase_Price_USD": 249.0, "TDP_Watts": 225, "Platform_Backend": "SYCL", "Max_Quant_Param_Size_B": 8.0},
        {"GPU_Name": "Arc A770 (16GB)", "VRAM_GB": 16, "Purchase_Price_USD": 349.0, "TDP_Watts": 225, "Platform_Backend": "SYCL", "Max_Quant_Param_Size_B": 16.0},
        # Specialized quantized architectures
        {"GPU_Name": "Bonsai 1 (8B 1-bit)", "VRAM_GB": 8, "Purchase_Price_USD": 1200.0, "TDP_Watts": 150, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 1.0},
        {"GPU_Name": "Ternary Bonsai 2 (27B 1.58-bit)", "VRAM_GB": 16, "Purchase_Price_USD": 2000.0, "TDP_Watts": 250, "Platform_Backend": "CUDA", "Max_Quant_Param_Size_B": 1.58},
    ]

    fieldnames = [
        "GPU_Name",
        "VRAM_GB",
        "Purchase_Price_USD",
        "TDP_Watts",
        "Platform_Backend",
        "Max_Quant_Param_Size_B",
    ]
    return save_csv(data_list, os.path.join(output_dir, "raw_gpu_hardware.csv"), fieldnames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("BI Data Extraction Script")
    print("Project: The Economics of AI: Cloud API Vendor Lock-In vs. Local On-Premise TCO")
    print("=" * 78)
    ensure_dir(DATA_RAW_DIR)

    results = {
        "openrouter": extract_openrouter_pricing(DATA_RAW_DIR),
        "huggingface": extract_huggingface_benchmarks(DATA_RAW_DIR),
        "gpu_hardware": generate_gpu_hardware(DATA_RAW_DIR),
    }

    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"  OpenRouter pricing   : {'SUCCESS' if results['openrouter'] else 'PARTIAL/FAILED'}")
    print(f"  HF benchmarks        : {'SUCCESS' if results['huggingface'] else 'PARTIAL/FAILED'}")
    print(f"  GPU hardware specs   : {'SUCCESS' if results['gpu_hardware'] else 'FAILED'}")
    print(f"  Output directory     : {os.path.abspath(DATA_RAW_DIR)}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
