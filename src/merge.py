"""
merge_sources.py
=================
Aligns two heterogeneous sources onto one common schema:

  - Newegg scrape (laptops_newegg.csv)      -> per-page provenance
    (scraped_at, html_sha256, field_provenance_json), real prices in USD,
    76/719 rows with genuine user reviews.

  - Kaggle "Laptop Price" dataset (laptop_price.csv, muhammetvarl)
    -> 1,303 rows, clean/no missing values, prices in EUR, NO reviews and
    NO battery field at all. Kaggle's own "last updated" metadata reads
    2020-11-05 -- treat that as the collection date; there is no page-level
    hash/provenance to attach because this is a static public dataset, not
    something we scraped ourselves. That asymmetry is recorded explicitly
    (source_dataset + source_collection_date + has_page_provenance) rather
    than papered over.

COMMON OUTPUT SCHEMA
---------------------
title, price_usd, price_original, price_original_currency, cpu, ram_gb,
storage, gpu, display, battery, category, document_text, has_review_text,
source_dataset, source_collection_date, source_row_id, has_page_provenance

KNOWN LIMITATIONS (documented, not hidden):
  - EUR->USD uses a single fixed rate (see EUR_TO_USD below) applied
    uniformly. The Kaggle prices reflect a ~2020 market; converting at
    today's exchange rate does NOT adjust for 5+ years of inflation/price
    change, so cross-source price comparisons are only roughly comparable.
    Flag this explicitly in the M1 report rather than presenting price_usd
    as apples-to-apples.
  - Kaggle rows have no battery spec at all -> combined corpus battery
    coverage will be even lower than the Newegg-only 46%. Report this,
    don't quietly impute it.
  - Kaggle rows have no review/user text -> document_text for those rows
    is a synthesized spec description, not organic text like the Newegg
    reviews. has_review_text distinguishes the two so retrieval/report
    logic doesn't treat them as equivalent evidence.

Run:
    python3 merge_sources.py laptops_newegg.csv laptop_price.csv combined_laptops.csv
"""

import re
import sys

import pandas as pd

EUR_TO_USD = 1.15  # mid-market rate, ~Aug 2026 — see note above re: staleness
KAGGLE_COLLECTION_DATE = "2020-11-05"  # per Kaggle dataset page metadata


def parse_ram(ram_str):
    if not isinstance(ram_str, str):
        return None
    match = re.search(r"(\d+)", ram_str)
    return int(match.group(1)) if match else None


def parse_storage(memory_str):
    """'128GB SSD +  1TB HDD' -> '128GB SSD + 1TB HDD' (just normalize
    whitespace; keep as descriptive string like the Newegg 'storage'
    field rather than forcing a single total-GB number, since type
    (SSD vs HDD vs Hybrid) matters for the RAG use case too)."""
    if not isinstance(memory_str, str):
        return None
    return re.sub(r"\s+", " ", memory_str).strip()


def kaggle_document_text(row):
    parts = [
        f"Product: {row['Company']} {row['Product']} ({row['TypeName']})",
        f"Price: EUR {row['Price_euros']:.2f}",
        f"Specifications: CPU: {row['Cpu']}; RAM: {row['Ram']}; "
        f"Storage: {row['Memory']}; GPU: {row['Gpu']}; "
        f"Display: {row['Inches']}\" {row['ScreenResolution']}; "
        f"OS: {row['OpSys']}; Weight: {row['Weight']}",
    ]
    return "\n".join(parts)


def load_newegg(path):
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "title": df["title"],
        "price_usd": pd.to_numeric(df["price"], errors="coerce"),
        "price_original": pd.to_numeric(df["price"], errors="coerce"),
        "price_original_currency": "USD",
        "cpu": df["cpu"],
        "ram_gb": df["ram"].apply(parse_ram) if "ram" in df else None,
        "storage": df["storage"],
        "gpu": df["gpu"],
        "display": df["display"],
        "battery": df["battery"],
        "category": df["category"],
        "document_text": df["document_text"] if "document_text" in df else df["description"],
        "has_review_text": df["reviews_json"].notna() & (df["reviews_json"].astype(str) != "[]")
                            if "reviews_json" in df else False,
        "source_dataset": "newegg_scrape",
        "source_collection_date": df["scraped_at"].str.slice(0, 10) if "scraped_at" in df else None,
        "source_row_id": df["url"],
        "has_page_provenance": True,
    })
    return out


def load_kaggle(path):
    df = pd.read_csv(path, encoding="latin-1")
    out = pd.DataFrame({
        "title": df["Company"] + " " + df["Product"],
        "price_usd": df["Price_euros"] * EUR_TO_USD,
        "price_original": df["Price_euros"],
        "price_original_currency": "EUR",
        "cpu": df["Cpu"],
        "ram_gb": df["Ram"].apply(parse_ram),
        "storage": df["Memory"].apply(parse_storage),
        "gpu": df["Gpu"],
        "display": df["Inches"].astype(str) + '" ' + df["ScreenResolution"],
        "battery": None,  # not present in this source — see docstring
        "category": df["TypeName"],
        "document_text": df.apply(kaggle_document_text, axis=1),
        "has_review_text": False,  # no organic review text in this source
        "source_dataset": "kaggle_laptop_price",
        "source_collection_date": KAGGLE_COLLECTION_DATE,
        "source_row_id": "kaggle_" + df["laptop_ID"].astype(str),
        "has_page_provenance": False,
    })
    return out


def main(newegg_path, kaggle_path, out_path):
    newegg = load_newegg(newegg_path)
    kaggle = load_kaggle(kaggle_path)
    combined = pd.concat([newegg, kaggle], ignore_index=True)

    print(f"Newegg rows:   {len(newegg)}")
    print(f"Kaggle rows:   {len(kaggle)}")
    print(f"Combined rows: {len(combined)} (pre-dedup — MinHash/LSH happens in Spark)")
    print()
    print("battery coverage in combined corpus: "
          f"{combined['battery'].notna().sum()}/{len(combined)} "
          f"({combined['battery'].notna().mean():.1%})")
    print("rows with genuine review text: "
          f"{combined['has_review_text'].sum()}/{len(combined)}")
    print()
    print("source_dataset breakdown:")
    print(combined["source_dataset"].value_counts())

    combined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    newegg_path = sys.argv[1] if len(sys.argv) > 1 else "laptops_newegg.csv"
    kaggle_path = sys.argv[2] if len(sys.argv) > 2 else "laptop_price.csv"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "combined_laptops.csv"
    main(newegg_path, kaggle_path, out_path)