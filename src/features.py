"""复购预测的特征工程。

思路：在某个快照日，用过去 180 天的行为构造特征，用未来 90 天是否购买构造标签。
两个快照分别做训练集和测试集，保证是时间外验证，而不是随机切分。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "total_items",
    "distinct_products",
    "avg_order_value",
    "avg_items_per_order",
    "avg_basket_lines",
    "tenure_days",
    "avg_gap_days",
    "purchase_days",
    "customer_age_days",
    "recency_ratio",
    "frequency_per_month",
    "monetary_per_day",
    "is_one_time",
    "active_months",
    "recent_90d_frequency",
    "recent_90d_monetary",
    "recent_90d_share",
    "cancel_count",
    "cancel_rate",
    "country_uk",
]


def load_sales() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.dropna(subset=["CustomerID"]).copy()


def load_cancellations() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "cancellations.parquet")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df.dropna(subset=["CustomerID"]).copy()


def build_snapshot(
    sales: pd.DataFrame,
    cancellations: pd.DataFrame,
    snapshot: str,
    observation_days: int = 180,
    label_days: int = 90,
) -> pd.DataFrame:
    """构造一个快照日的客户特征与未来复购标签。"""
    snapshot_ts = pd.Timestamp(snapshot)
    obs_start = snapshot_ts - pd.Timedelta(days=observation_days)
    label_end = snapshot_ts + pd.Timedelta(days=label_days)

    obs = sales[
        (sales["InvoiceDate"] > obs_start)
        & (sales["InvoiceDate"] <= snapshot_ts)
    ].copy()
    obs = obs.sort_values("InvoiceDate")
    future = sales[
        (sales["InvoiceDate"] > snapshot_ts)
        & (sales["InvoiceDate"] <= label_end)
    ]

    if obs.empty:
        raise ValueError(f"快照 {snapshot} 之前没有可用销售数据")

    grouped = obs.groupby("CustomerID", observed=True)
    features = grouped.agg(
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
        total_items=("Quantity", "sum"),
        distinct_products=("StockCode", "nunique"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        basket_lines=("InvoiceNo", "size"),
        purchase_days=("InvoiceDateOnly", "nunique"),
        active_months=("InvoiceMonth", "nunique"),
        country=("Country", "last"),
    ).reset_index()

    features["recency_days"] = (
        snapshot_ts - features["last_purchase"]
    ).dt.days
    features["tenure_days"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days
    features["avg_order_value"] = features["monetary"] / features["frequency"]
    features["avg_items_per_order"] = (
        features["total_items"] / features["frequency"]
    )
    features["avg_basket_lines"] = features["basket_lines"] / features["frequency"]
    features["avg_gap_days"] = np.where(
        features["frequency"] > 1,
        features["tenure_days"] / (features["frequency"] - 1),
        features["tenure_days"],
    )
    features["customer_age_days"] = (
        snapshot_ts - features["first_purchase"]
    ).dt.days
    features["recency_ratio"] = features["recency_days"] / (
        features["avg_gap_days"] + 1
    )
    features["frequency_per_month"] = (
        features["frequency"] / features["active_months"]
    )
    features["monetary_per_day"] = (
        features["monetary"] / features["customer_age_days"].clip(lower=1)
    )
    features["is_one_time"] = (features["frequency"] == 1).astype(int)

    recent_start = snapshot_ts - pd.Timedelta(days=90)
    recent = (
        obs[obs["InvoiceDate"] > recent_start]
        .groupby("CustomerID", observed=True)
        .agg(
            recent_90d_frequency=("InvoiceNo", "nunique"),
            recent_90d_monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    features = features.merge(recent, on="CustomerID", how="left")
    features[["recent_90d_frequency", "recent_90d_monetary"]] = features[
        ["recent_90d_frequency", "recent_90d_monetary"]
    ].fillna(0)
    features["recent_90d_share"] = np.where(
        features["monetary"] > 0,
        features["recent_90d_monetary"] / features["monetary"],
        0,
    )

    cancel = cancellations[
        (cancellations["InvoiceDate"] > obs_start)
        & (cancellations["InvoiceDate"] <= snapshot_ts)
    ]
    cancel_count = (
        cancel.groupby("CustomerID", observed=True)["InvoiceNo"]
        .nunique()
        .rename("cancel_count")
        .reset_index()
    )
    features = features.merge(cancel_count, on="CustomerID", how="left")
    features["cancel_count"] = features["cancel_count"].fillna(0)
    features["cancel_rate"] = features["cancel_count"] / (
        features["cancel_count"] + features["frequency"]
    )
    features["country_uk"] = (
        features["country"].astype(str) == "United Kingdom"
    ).astype(int)

    future_customers = set(future["CustomerID"].dropna().unique())
    features["label"] = features["CustomerID"].isin(future_customers).astype(int)
    features["snapshot"] = snapshot_ts
    features["observation_start"] = obs_start
    features["label_end"] = label_end
    features["customer_id"] = features["CustomerID"]
    features = features.drop(columns=["CustomerID"])

    return features


def build_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """训练快照 2011-06-09，测试快照 2011-09-09，各自看未来 90 天。"""
    sales = load_sales()
    cancellations = load_cancellations()
    train = build_snapshot(sales, cancellations, "2011-06-09")
    test = build_snapshot(sales, cancellations, "2011-09-09")
    return train, test


if __name__ == "__main__":
    train, test = build_train_test()
    print(f"训练集: {train.shape[0]:,} 客户, 复购率 {train['label'].mean():.2%}")
    print(f"测试集: {test.shape[0]:,} 客户, 复购率 {test['label'].mean():.2%}")
    print(f"特征数: {len(FEATURE_COLUMNS)}")
    print(train[FEATURE_COLUMNS].describe().T.round(3).to_string())
