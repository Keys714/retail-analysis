"""电商零售分析与复购预测看板。

运行：
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"
SURFACE = "#fcfcfb"

st.set_page_config(
    page_title="电商零售分析与复购预测",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_sales() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_rfm() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "customer_rfm.parquet")


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "repurchase_test_scores.parquet")


@st.cache_data
def load_json(name: str) -> dict:
    with open(REPORTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def kpi_values(df: pd.DataFrame) -> dict:
    revenue = float(df["Revenue"].sum())
    orders = int(df["InvoiceNo"].nunique())
    customers = int(df["CustomerID"].nunique())
    cust = (
        df.dropna(subset=["CustomerID"])
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
    )
    repeat = float((cust >= 2).mean() * 100) if len(cust) else 0.0
    return {
        "revenue": revenue,
        "orders": orders,
        "customers": customers,
        "aov": revenue / orders if orders else 0,
        "repeat": repeat,
    }


def monthly_chart(df: pd.DataFrame) -> alt.Chart:
    monthly = (
        df.groupby("InvoiceMonth", as_index=False)["Revenue"]
        .sum()
        .sort_values("InvoiceMonth")
    )
    return (
        alt.Chart(monthly)
        .mark_line(color=BLUE, strokeWidth=2)
        .encode(
            x=alt.X(
                "InvoiceMonth:N",
                title=None,
                axis=alt.Axis(labelAngle=-45, labelLimit=90),
            ),
            y=alt.Y(
                "Revenue:Q",
                title="商品销售额",
                axis=alt.Axis(format="~s"),
            ),
            tooltip=[
                alt.Tooltip("InvoiceMonth:N", title="月份"),
                alt.Tooltip("Revenue:Q", title="销售额", format=",.0f"),
            ],
        )
        .properties(height=330, title="月度商品销售额")
    )


def country_chart(df: pd.DataFrame) -> alt.Chart:
    country = (
        df.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    return (
        alt.Chart(country)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("Revenue:Q", title="商品销售额", axis=alt.Axis(format="~s")),
            y=alt.Y("Country:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Country:N", title="国家/地区"),
                alt.Tooltip("Revenue:Q", title="销售额", format=",.0f"),
            ],
        )
        .properties(height=330, title="销售额 Top 10 国家/地区")
    )


def rfm_chart(rfm: pd.DataFrame) -> alt.Chart:
    seg = (
        rfm.groupby("客户分层", as_index=False)
        .agg(客户数=("CustomerID", "size"), 销售额=("销售额", "sum"))
    )
    seg["销售额占比"] = seg["销售额"] / seg["销售额"].sum() * 100
    seg["客户数占比"] = seg["客户数"] / seg["客户数"].sum() * 100
    long = seg.melt(
        id_vars="客户分层",
        value_vars=["销售额占比", "客户数占比"],
        var_name="指标",
        value_name="占比",
    )
    order = (
        seg.sort_values("销售额占比", ascending=False)["客户分层"].tolist()
    )
    return (
        alt.Chart(long)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("客户分层:N", sort=order, title=None),
            x=alt.X("占比:Q", title="占比（%）", axis=alt.Axis(format=".0f")),
            yOffset="指标:N",
            color=alt.Color(
                "指标:N",
                scale=alt.Scale(
                    domain=["销售额占比", "客户数占比"],
                    range=[BLUE, ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("客户分层:N", title="客户分层"),
                alt.Tooltip("指标:N", title="指标"),
                alt.Tooltip("占比:Q", title="占比", format=".1f"),
            ],
        )
        .properties(height=330, title="RFM 客户分层（快照 2011-12-10）")
    )


sales = load_sales()
rfm = load_rfm()
scores = load_scores()
cleaning = load_json("cleaning_summary.json")
model = load_json("model_metrics.json")

st.title("电商零售用户分析与复购预测")
st.caption(
    "UCI Online Retail · 541,909 条交易明细 · 2010-12 至 2011-12 · "
    "有效商品销售口径（不含邮费/手续费）"
)

min_date = sales["InvoiceDate"].min().date()
max_date = sales["InvoiceDate"].max().date()
country_options = (
    sales.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)
default_countries = country_options

with st.sidebar:
    st.header("筛选条件")
    date_range = st.date_input(
        "日期范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    countries = st.multiselect(
        "国家/地区",
        options=country_options,
        default=default_countries,
    )
    st.caption("默认展示全部国家/地区；可多选或清空后重新选择。")
    st.divider()
    st.markdown("**数据口径**")
    st.caption(
        "销售额 = Quantity × UnitPrice；仅保留数量 > 0、单价 > 0、"
        "非取消发票、非邮费/手续费等非商品编码的记录。"
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (sales["InvoiceDate"].dt.date >= start_date)
    & (sales["InvoiceDate"].dt.date <= end_date)
    & (sales["Country"].isin(countries))
)
filtered = sales.loc[mask].copy()

if filtered.empty:
    st.warning("当前筛选条件下没有数据，请调整日期或国家。")
    st.stop()

kpis = kpi_values(filtered)
cols = st.columns(6)
cols[0].metric("商品销售额", f"{kpis['revenue']/10000:,.1f} 万")
cols[1].metric("订单数", f"{kpis['orders']:,}")
cols[2].metric("有 ID 客户数", f"{kpis['customers']:,}")
cols[3].metric("客单价", f"{kpis['aov']:,.2f}")
cols[4].metric("复购率", f"{kpis['repeat']:.1f}%")
cols[5].metric(
    "退货率（金额）",
    f"{cleaning['quality']['return_rate_revenue_pct']:.2f}%",
)

st.caption(
    f"当前筛选：{start_date} 至 {end_date}，"
    f"{len(countries)} 个国家/地区；退货率为全量口径。"
)

st.divider()

st.altair_chart(monthly_chart(filtered), width="stretch")
st.caption("注：2011-12 数据仅到 12 月 9 日，最后一个月为不完整月份。")

left, right = st.columns(2)
with left:
    st.altair_chart(country_chart(filtered), width="stretch")
with right:
    st.altair_chart(rfm_chart(rfm), width="stretch")

st.divider()
st.subheader("复购风险名单（回测）")
st.caption(
    "基于快照日 2011-09-09 的特征，预测未来 90 天是否复购。"
    "概率越低越需要召回；实际标签仅用于模型评估，不在名单中展示。"
)

risk_left, risk_right = st.columns([1, 1])
with risk_left:
    max_prob = st.slider(
        "复购概率低于",
        min_value=0.05,
        max_value=0.80,
        value=0.30,
        step=0.05,
    )
with risk_right:
    min_value = st.number_input(
        "最低 180 天消费额",
        min_value=0,
        value=1000,
        step=500,
    )

risk = scores[
    (scores["probability"] <= max_prob)
    & (scores["monetary"] >= min_value)
].sort_values("monetary", ascending=False).head(20).copy()

if risk.empty:
    st.info("当前条件下没有符合条件的客户，可放宽概率或降低消费额门槛。")
else:
    risk["复购概率"] = (risk["probability"] * 100).round(1)
    risk_view = risk[
        [
            "customer_id",
            "recency_days",
            "frequency",
            "monetary",
            "复购概率",
        ]
    ].rename(
        columns={
            "customer_id": "客户 ID",
            "recency_days": "最近购买间隔（天）",
            "frequency": "180 天购买次数",
            "monetary": "180 天消费额",
            "复购概率": "复购概率（%）",
        }
    )
    st.dataframe(
        risk_view,
        hide_index=True,
        width="stretch",
        column_config={
            "180 天消费额": st.column_config.NumberColumn(format="%.0f"),
            "复购概率（%）": st.column_config.NumberColumn(format="%.1f"),
        },
    )

with st.expander("模型评估与口径说明"):
    best = next(
        r for r in model["results"] if r["model"] == model["best_model"]
    )
    st.markdown(
        f"""
- 最佳模型：**{model['best_model']}**
- 训练快照：2011-06-09（{model['train']['customers']:,} 人，复购率 {model['train']['positive_rate']:.1%}）
- 测试快照：2011-09-09（{model['test']['customers']:,} 人，复购率 {model['test']['positive_rate']:.1%}）
- 交叉验证 AUC：**{best['cv_auc_mean']:.3f}**
- 时间外测试 AUC：**{best['roc_auc']:.3f}**
- Top 10% 精确率：**{best['top_10pct']['precision']:.1%}**，Lift **{best['top_10pct']['lift']:.2f}**
- 特征窗口：快照日前 180 天；标签窗口：快照日后 90 天
- RFM 快照日：2011-12-10
        """
    )

st.divider()
st.caption(
    "数据来源：UCI Machine Learning Repository — Online Retail。"
    "本项目仅用于学习与作品集展示，结论不代表真实企业经营情况。"
)
