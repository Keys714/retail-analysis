"""RFM 客户分层。

R（Recency）：距离快照日的最近购买天数，越小越好。
F（Frequency）：统计期内去重发票数。
M（Monetary）：统计期内商品销售额。

每个维度用分位数打 1-5 分，再按规则映射到客户分层。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def assign_segment(row: pd.Series) -> str:
    r, f, m = row["R分"], row["F分"], row["M分"]
    if r >= 4 and f >= 4 and m >= 4:
        return "高价值忠诚客户"
    if r >= 3 and f >= 3:
        return "稳定复购客户"
    if r >= 4 and f <= 2:
        return "新客/潜力客户"
    if r <= 2 and f >= 3:
        return "流失预警客户"
    if r <= 2 and f <= 2:
        return "沉睡客户"
    return "一般客户"


def build_rfm() -> pd.DataFrame:
    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    sales = sales.dropna(subset=["CustomerID"])

    snapshot = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    customer = (
        sales.groupby("CustomerID")
        .agg(
            订单数=("InvoiceNo", "nunique"),
            销售额=("Revenue", "sum"),
            首次购买=("InvoiceDate", "min"),
            最近购买=("InvoiceDate", "max"),
        )
        .reset_index()
    )

    customer["R值"] = (snapshot - customer["最近购买"]).dt.days
    customer["F值"] = customer["订单数"]
    customer["M值"] = customer["销售额"]

    customer["R分"] = pd.qcut(
        customer["R值"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    customer["F分"] = pd.qcut(
        customer["F值"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    customer["M分"] = pd.qcut(
        customer["M值"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    customer["RFM综合分"] = customer["R分"] + customer["F分"] + customer["M分"]
    customer["客户分层"] = customer.apply(assign_segment, axis=1)
    customer["快照日"] = snapshot

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customer.to_parquet(PROCESSED_DIR / "customer_rfm.parquet", index=False)
    return customer


if __name__ == "__main__":
    rfm = build_rfm()
    summary = (
        rfm.groupby("客户分层")
        .agg(客户数=("CustomerID", "size"), 销售额=("销售额", "sum"))
        .sort_values("销售额", ascending=False)
    )
    summary["客户占比%"] = (summary["客户数"] / len(rfm) * 100).round(2)
    summary["销售额占比%"] = (
        summary["销售额"] / summary["销售额"].sum() * 100
    ).round(2)
    print(f"RFM 快照日: {rfm['快照日'].iloc[0].date()}; 客户数: {len(rfm):,}")
    print(summary.round(2).to_string())
