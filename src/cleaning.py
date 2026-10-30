"""数据清洗模块。

输入：data/raw/Online Retail.xlsx
输出：
- data/processed/sales.parquet          有效商品销售明细（分析主表）
- data/processed/cancellations.parquet  取消/退货发票
- data/processed/adjustments.parquet    非 C 发票的负数量库存调整
- data/processed/service_fees.parquet   邮费、运费、手续费等非商品收入
- data/processed/clean_all.parquet      去重后的全量明细（带标记）
- reports/cleaning_summary.json         清洗前后数量与口径统计
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_EXCEL = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
RAW_PARQUET = PROJECT_ROOT / "data" / "processed" / "raw.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 数据集中已知的非商品编码：邮费、运费、人工调整、折扣、手续费等
SERVICE_CODES = {
    "POST",          # POSTAGE
    "DOT",           # DOTCOM POSTAGE
    "M",             # Manual
    "C2",            # CARRIAGE
    "D",             # Discount
    "S",             # SAMPLES
    "B",             # Adjust bad debt
    "PADS",          # PADS TO MATCH ALL CUSHIONS
    "CRUK",          # CRUK Commission
    "BANK CHARGES",  # Bank Charges
    "AMAZONFEE",     # AMAZON FEE
}

TEXT_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Country"]


def load_raw() -> pd.DataFrame:
    """优先读取 Parquet 缓存，没有则从 Excel 读取并缓存。"""
    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)

    df = pd.read_excel(RAW_EXCEL)
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string")
    df.to_parquet(RAW_PARQUET, index=False)
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一字段类型，并补充时间与金额字段。"""
    df = df.copy()

    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("string")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDateOnly"] = df["InvoiceDate"].dt.normalize()
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """给每一行打上取消、服务费、有效销售等标记。"""
    df = df.copy()

    invoice_str = df["InvoiceNo"].fillna("").astype(str)
    stock_str = df["StockCode"].fillna("").astype(str)

    df["IsCancelled"] = invoice_str.str.startswith("C")
    df["IsService"] = stock_str.isin(SERVICE_CODES)
    df["IsAdjustment"] = (df["Quantity"] < 0) & ~df["IsCancelled"]
    df["IsValidSale"] = (
        (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & ~df["IsCancelled"]
    )
    return df


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按业务口径拆分出四张分析表。"""
    return {
        "sales": df[df["IsValidSale"] & ~df["IsService"]].copy(),
        "service_fees": df[df["IsValidSale"] & df["IsService"]].copy(),
        "cancellations": df[df["IsCancelled"]].copy(),
        "adjustments": df[df["IsAdjustment"]].copy(),
        "clean_all": df.copy(),
    }


def summarize(
    raw_rows: int,
    duplicate_rows: int,
    deduped: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> dict:
    """计算清洗前后规模、退货率、匿名交易占比等指标。"""
    sales = tables["sales"]
    cancellations = tables["cancellations"]
    service_fees = tables["service_fees"]
    adjustments = tables["adjustments"]

    # 商品退货：只统计非服务费编码的取消发票
    cancel_product = cancellations[~cancellations["IsService"]]
    cancel_revenue_abs = abs(float(cancel_product["Revenue"].sum()))
    sales_revenue = float(sales["Revenue"].sum())

    valid_invoices = set(sales["InvoiceNo"].dropna())
    cancelled_invoices = set(cancellations["InvoiceNo"].dropna())

    return {
        "raw_rows": int(raw_rows),
        "duplicate_rows": int(duplicate_rows),
        "rows_after_dedup": int(len(deduped)),
        "sales": {
            "rows": int(len(sales)),
            "invoices": int(sales["InvoiceNo"].nunique()),
            "customers": int(sales["CustomerID"].nunique()),
            "products": int(sales["StockCode"].nunique()),
            "countries": int(sales["Country"].nunique()),
            "revenue": round(sales_revenue, 2),
            "date_min": str(sales["InvoiceDate"].min()),
            "date_max": str(sales["InvoiceDate"].max()),
        },
        "cancellations": {
            "rows": int(len(cancellations)),
            "invoices": int(cancellations["InvoiceNo"].nunique()),
            "revenue": round(float(cancellations["Revenue"].sum()), 2),
            "product_revenue_abs": round(cancel_revenue_abs, 2),
        },
        "adjustments": {
            "rows": int(len(adjustments)),
            "invoices": int(adjustments["InvoiceNo"].nunique()),
        },
        "service_fees": {
            "rows": int(len(service_fees)),
            "revenue": round(float(service_fees["Revenue"].sum()), 2),
            "codes": sorted(service_fees["StockCode"].dropna().unique().tolist()),
        },
        "quality": {
            "missing_customer_rows": int(deduped["CustomerID"].isna().sum()),
            "missing_customer_rate_pct": round(
                float(deduped["CustomerID"].isna().mean() * 100), 2
            ),
            "anonymous_sales_share_pct": round(
                float(
                    sales.loc[sales["CustomerID"].isna(), "Revenue"].sum()
                    / sales_revenue
                    * 100
                ),
                2,
            ),
            "non_positive_price_rows": int((deduped["UnitPrice"] <= 0).sum()),
            "return_rate_revenue_pct": round(
                cancel_revenue_abs / (sales_revenue + cancel_revenue_abs) * 100, 2
            ),
            "cancelled_invoice_rate_pct": round(
                len(cancelled_invoices)
                / (len(valid_invoices | cancelled_invoices))
                * 100,
                2,
            ),
        },
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    raw_rows = len(raw)

    norm = normalize(raw)
    duplicate_rows = int(norm.duplicated().sum())
    deduped = norm.drop_duplicates().copy()

    flagged = add_flags(deduped)
    tables = build_tables(flagged)

    for name, table in tables.items():
        table.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    summary = summarize(raw_rows, duplicate_rows, deduped, tables)
    with open(REPORTS_DIR / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
