"""业务分析、RFM 分层、图表与报告生成。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 来自 dataviz 技能参考调色板的已验证默认值
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
ORANGE = "#eb6834"
GRAY = "#c3c2b7"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    sales = pd.read_parquet(PROCESSED_DIR / "sales.parquet")
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    sales["InvoiceMonth"] = sales["InvoiceDate"].dt.to_period("M").astype(str)

    rfm_path = PROCESSED_DIR / "customer_rfm.parquet"
    if not rfm_path.exists():
        from src.rfm import build_rfm

        build_rfm()
    rfm = pd.read_parquet(rfm_path)

    with open(REPORTS_DIR / "cleaning_summary.json", encoding="utf-8") as f:
        cleaning = json.load(f)
    with open(REPORTS_DIR / "model_metrics.json", encoding="utf-8") as f:
        model = json.load(f)
    return sales, rfm, cleaning, model


def compute_kpis(sales: pd.DataFrame, rfm: pd.DataFrame, cleaning: dict) -> dict:
    revenue = float(sales["Revenue"].sum())
    orders = int(sales["InvoiceNo"].nunique())
    customers = int(sales["CustomerID"].nunique())
    products = int(sales["StockCode"].nunique())

    customer = (
        sales.dropna(subset=["CustomerID"])
        .groupby("CustomerID")
        .agg(订单数=("InvoiceNo", "nunique"), 销售额=("Revenue", "sum"))
    )
    top10 = (
        customer["销售额"].sort_values(ascending=False).head(int(len(customer) * 0.1)).sum()
        / customer["销售额"].sum()
        * 100
    )
    repeat = (customer["订单数"] >= 2).mean() * 100

    country = sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    uk_share = country.get("United Kingdom", 0) / revenue * 100

    product = sales.groupby("StockCode")["Revenue"].sum().sort_values(ascending=False)
    n20 = max(int(np.ceil(len(product) * 0.2)), 1)
    top20_product = product.iloc[:n20].sum() / revenue * 100

    return {
        "revenue": revenue,
        "orders": orders,
        "customers": customers,
        "products": products,
        "aov": revenue / orders,
        "revenue_per_customer": revenue / customers,
        "avg_items_per_order": float(sales["Quantity"].sum()) / orders,
        "avg_lines_per_order": len(sales) / orders,
        "repeat_rate_pct": repeat,
        "top10_customer_share_pct": top10,
        "uk_share_pct": uk_share,
        "top20_product_share_pct": top20_product,
        "return_rate_pct": cleaning["quality"]["return_rate_revenue_pct"],
        "cancelled_invoice_rate_pct": cleaning["quality"]["cancelled_invoice_rate_pct"],
        "anonymous_sales_share_pct": cleaning["quality"]["anonymous_sales_share_pct"],
        "date_min": str(sales["InvoiceDate"].min()),
        "date_max": str(sales["InvoiceDate"].max()),
    }


def plot_monthly(sales: pd.DataFrame) -> None:
    monthly = (
        sales.groupby("InvoiceMonth")["Revenue"].sum().sort_index().reset_index()
    )
    monthly = monthly[monthly["InvoiceMonth"] < "2011-12"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(monthly))
    y = monthly["Revenue"].to_numpy()
    ax.plot(x, y, color=BLUE, linewidth=2, zorder=3)
    ax.fill_between(x, y, color=BLUE_LIGHT, alpha=0.45, zorder=2)

    peak = int(np.argmax(y))
    ax.scatter([x[peak]], [y[peak]], s=50, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.annotate(
        f"峰值 {y[peak]/10000:.1f} 万",
        xy=(x[peak], y[peak]),
        xytext=(x[peak] - 2.1, y[peak] * 0.93),
        fontsize=9,
        color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(monthly["InvoiceMonth"], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/10000:.0f}万"))
    ax.set_ylabel("商品销售额")
    ax.set_title(
        "月度商品销售额（2010-12 至 2011-11）",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
        pad=18,
    )
    ax.text(
        0.01,
        0.97,
        "注：2011-12 数据仅到 12 月 9 日，未纳入趋势",
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="top",
    )
    style_axes(ax)
    save(fig, "monthly_revenue.png")


def plot_country(sales: pd.DataFrame, uk_share: float) -> None:
    country = (
        sales.groupby("Country")["Revenue"].sum().sort_values(ascending=False) / sales["Revenue"].sum() * 100
    )
    non_uk = country.drop("United Kingdom", errors="ignore").head(8)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 2.1]}
    )
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    vals = [uk_share, 100 - uk_share]
    bars = ax.bar(["英国", "其他市场"], vals, color=[BLUE, GRAY], width=0.55)
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
            color=INK,
        )
    ax.set_ylim(0, 100)
    ax.set_title("英国 vs 其他市场", loc="left", fontsize=11, color=INK)
    style_axes(ax)

    ax = axes[1]
    labels = non_uk.index.tolist()[::-1]
    values = non_uk.to_numpy()[::-1]
    ax.barh(labels, values, color=BLUE, height=0.55)
    for i, value in enumerate(values):
        ax.text(value + 0.05, i, f"{value:.1f}%", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_title("其他市场 Top 8（销售额占比）", loc="left", fontsize=11, color=INK)
    style_axes(ax, grid_axis="x")

    fig.suptitle(
        f"市场集中度：英国贡献 {uk_share:.1f}% 销售额",
        x=0.01,
        y=1.03,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    save(fig, "country_concentration.png")


def plot_rfm(rfm: pd.DataFrame) -> None:
    seg = (
        rfm.groupby("客户分层")
        .agg(客户数=("CustomerID", "size"), 销售额=("销售额", "sum"))
        .sort_values("销售额", ascending=True)
    )
    total_customers = seg["客户数"].sum()
    total_revenue = seg["销售额"].sum()
    seg["客户占比"] = seg["客户数"] / total_customers * 100
    seg["销售额占比"] = seg["销售额"] / total_revenue * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(seg))
    h = 0.36
    ax.barh(y + h / 2, seg["销售额占比"], height=h, color=BLUE, label="销售额占比")
    ax.barh(y - h / 2, seg["客户占比"], height=h, color=ORANGE, label="客户数占比")

    for i, (rev, cust) in enumerate(zip(seg["销售额占比"], seg["客户占比"])):
        ax.text(rev + 0.8, i + h / 2, f"{rev:.1f}%", va="center", fontsize=9, color=INK_SECONDARY)
        ax.text(cust + 0.8, i - h / 2, f"{cust:.1f}%", va="center", fontsize=9, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(seg.index)
    ax.set_xlim(0, max(seg["销售额占比"].max(), seg["客户占比"].max()) * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(
        "RFM 客户分层：高价值忠诚客户贡献 64.8% 销售额",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
        pad=16,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax, grid_axis="x")
    save(fig, "rfm_segments.png")


def plot_roc() -> None:
    roc = pd.read_csv(REPORTS_DIR / "roc_curve.csv")
    scores = pd.read_parquet(PROCESSED_DIR / "repurchase_test_scores.parquet")
    fpr_base, tpr_base, _ = roc_curve(scores["label"], -scores["recency_days"])
    auc_base = roc_auc_score(scores["label"], -scores["recency_days"])
    auc_model = roc_auc_score(scores["label"], scores["probability"])

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=AXIS, linewidth=1.2, label="随机猜测")
    ax.plot(
        fpr_base,
        tpr_base,
        color=ORANGE,
        linewidth=2,
        label=f"近期购买规则 (AUC {auc_base:.3f})",
    )
    ax.plot(
        roc["fpr"],
        roc["tpr"],
        color=BLUE,
        linewidth=2,
        label=f"复购模型 (AUC {auc_model:.3f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("假阳性率（FPR）")
    ax.set_ylabel("真阳性率（TPR）")
    ax.set_title(
        "复购预测 ROC 曲线（时间外验证）",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
        pad=14,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax, grid_axis="both")
    save(fig, "roc_curve.png")


def plot_importance() -> None:
    imp = pd.read_csv(REPORTS_DIR / "feature_importance.csv").head(10)
    label_map = {
        "purchase_days": "购买天数",
        "frequency": "购买次数",
        "tenure_days": "活跃跨度（天）",
        "avg_basket_lines": "平均每单行数",
        "recent_90d_frequency": "近90天购买次数",
        "avg_gap_days": "平均购买间隔（天）",
        "recency_days": "最近购买距今天数",
        "avg_order_value": "平均客单价",
        "is_one_time": "是否一次性客户",
        "cancel_count": "取消发票数",
        "recency_ratio": "复购周期比",
        "active_months": "活跃月份数",
    }
    imp["label"] = imp["feature"].map(label_map).fillna(imp["feature"])
    imp = imp.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(imp["label"], imp["importance_mean"], color=BLUE, height=0.55)
    for i, value in enumerate(imp["importance_mean"]):
        ax.text(value + 0.004, i, f"{value:.3f}", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, imp["importance_mean"].max() * 1.18)
    ax.set_title(
        "复购模型特征重要性 Top 10（置换重要性，AUC）",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
        pad=14,
    )
    style_axes(ax, grid_axis="x")
    save(fig, "feature_importance.png")


def write_report(sales, rfm, cleaning, model, kpis) -> None:
    country = (
        sales.groupby("Country")
        .agg(销售额=("Revenue", "sum"), 订单数=("InvoiceNo", "nunique"))
        .sort_values("销售额", ascending=False)
    )
    country["销售额占比"] = country["销售额"] / kpis["revenue"] * 100

    product = (
        sales.groupby(["StockCode", "Description"], dropna=False)
        .agg(销售额=("Revenue", "sum"), 销量=("Quantity", "sum"), 订单数=("InvoiceNo", "nunique"))
        .sort_values("销售额", ascending=False)
        .head(5)
    )

    seg = (
        rfm.groupby("客户分层")
        .agg(客户数=("CustomerID", "size"), 销售额=("销售额", "sum"))
        .sort_values("销售额", ascending=False)
    )
    seg["客户占比"] = seg["客户数"] / seg["客户数"].sum() * 100
    seg["销售额占比"] = seg["销售额"] / seg["销售额"].sum() * 100

    best = next(r for r in model["results"] if r["model"] == model["best_model"])
    baseline = next(r for r in model["results"] if r["model"] == "基线：近期购买规则")
    gb = next(r for r in model["results"] if r["model"] == "梯度提升树")

    country_rows = "\n".join(
        f"| {idx} | {row['销售额']:,.0f} | {row['销售额占比']:.2f}% | {int(row['订单数']):,} |"
        for idx, row in country.head(5).iterrows()
    )
    rfm_rows = "\n".join(
        f"| {idx} | {int(row['客户数']):,} | {row['客户占比']:.2f}% | {row['销售额']:,.0f} | {row['销售额占比']:.2f}% |"
        for idx, row in seg.iterrows()
    )
    product_rows = "\n".join(
        f"| {idx[1]} | {row['销售额']:,.0f} | {row['销量']:,.0f} | {int(row['订单数']):,} |"
        for idx, row in product.iterrows()
    )

    text = f"""# 电商零售数据分析与复购预测报告

## 1. 项目概述

基于 UCI Online Retail 公开数据集（2010-12 至 2011-12，541,909 行交易明细），完成数据清洗、销售分析、RFM 客户分层与复购预测，并交付 Streamlit 交互看板。

**分析口径**：销售额 = `Quantity × UnitPrice`；仅统计数量 > 0、单价 > 0、非取消发票、非邮费/手续费等非商品编码的有效商品销售。

## 2. 数据清洗

| 指标 | 数值 |
|---|---:|
| 原始行数 | {cleaning['raw_rows']:,} |
| 完全重复行 | {cleaning['duplicate_rows']:,} |
| 去重后行数 | {cleaning['rows_after_dedup']:,} |
| 有效商品销售行数 | {cleaning['sales']['rows']:,} |
| 有效订单数 | {cleaning['sales']['invoices']:,} |
| 有 ID 客户数 | {cleaning['sales']['customers']:,} |
| 商品数 | {cleaning['sales']['products']:,} |
| 商品销售额 | {cleaning['sales']['revenue']:,.2f} |
| 取消发票数 | {cleaning['cancellations']['invoices']:,} |
| 库存调整行数 | {cleaning['adjustments']['rows']:,} |
| 邮费等非商品收入 | {cleaning['service_fees']['revenue']:,.2f} |

清洗要点：去重；区分正常销售、取消/退货、库存调整；把 `POST`、`DOT`、`M`、`C2`、`D`、`S`、`BANK CHARGES`、`AMAZONFEE`、`CRUK`、`PADS`、`B` 等非商品编码单独统计，避免把邮费误判为热销商品。

## 3. 核心发现

### 3.1 销售规模与季节性

- 商品销售额 **{kpis['revenue']/10000:,.1f} 万**，订单 **{kpis['orders']:,}** 笔，有 ID 客户 **{kpis['customers']:,}** 人。
- 客单价 **{kpis['aov']:,.2f}**，人均消费 **{kpis['revenue_per_customer']:,.2f}**，平均每单 **{kpis['avg_lines_per_order']:.1f}** 行商品明细（金额受大额批发单影响）。
- 销售额从 9 月开始快速上升，**11 月达到峰值 145.2 万**，呈现明显的 Q4 旺季特征；12 月数据仅到 9 日，未纳入趋势判断。

![月度销售额](figures/monthly_revenue.png)

### 3.2 市场高度集中

- 英国贡献 **{kpis['uk_share_pct']:.2f}%** 销售额，是绝对核心市场。
- 其他主要市场：荷兰 2.77%、爱尔兰 2.64%、德国 2.00%、法国 1.80%。
- 结论：短期应守住英国基本盘；中期优先测试荷兰、爱尔兰、德国、法国四个市场，而不是平均铺开。

![国家集中度](figures/country_concentration.png)

| 国家 | 销售额 | 占比 | 订单数 |
|---|---:|---:|---:|
{country_rows}

### 3.3 商品集中度高，但需警惕一次性大单

- 商品数 **{kpis['products']:,}** 个；**Top 20% 商品贡献 {kpis['top20_product_share_pct']:.2f}%** 销售额。
- 销售额 Top 5 商品：

| 商品 | 销售额 | 销量 | 订单数 |
|---|---:|---:|---:|
{product_rows}

- 注意：`PAPER CRAFT, LITTLE BIRDIE` 的 16.8 万销售额来自 **1 笔订单**，说明收入排名会被一次性批发单放大。做商品策略时应同时看订单数和复购，而不是只看销售额。

### 3.4 客户价值分化明显

- 有 ID 客户 **{kpis['customers']:,}** 人，复购率 **{kpis['repeat_rate_pct']:.2f}%**。
- **Top 10% 客户贡献 {kpis['top10_customer_share_pct']:.2f}%** 销售额，客户集中度高。
- 匿名交易占全部交易行的 25.16%，但只贡献 14.74% 销售额，说明匿名订单金额相对较小。

![RFM 分层](figures/rfm_segments.png)

| 客户分层 | 客户数 | 客户占比 | 销售额 | 销售额占比 |
|---|---:|---:|---:|---:|
{rfm_rows}

运营含义：

- **高价值忠诚客户**：21.71% 的人数贡献 64.78% 销售额，优先做会员权益、专属客服和稳定复购激励。
- **流失预警客户**：657 人、贡献 9.30% 销售额，最近购买间隔已明显变长，适合定向召回。
- **新客/潜力客户**：310 人，客单价偏低但最近有购买，适合用第二单优惠提升复购。
- **沉睡客户**：1,077 人，占人数最多但只贡献 5.90% 销售额，召回成本高，应控制投入。

### 3.5 退货与数据质量

- 商品退货率（金额口径）**{kpis['return_rate_pct']:.2f}%**；取消发票占全部发票的 **{kpis['cancelled_invoice_rate_pct']:.2f}%**。
- 退货率整体可控，但取消发票占比不低，建议按商品、国家、客户三类维度拆解取消原因，优先治理高频退货商品。

## 4. 复购预测模型

### 4.1 方法

- **特征窗口**：快照日前 180 天的行为。
- **标签窗口**：快照日后 90 天内是否再次购买。
- **训练集**：快照 2011-06-09，{model['train']['customers']:,} 人，复购率 {model['train']['positive_rate']:.2%}。
- **测试集**：快照 2011-09-09，{model['test']['customers']:,} 人，复购率 {model['test']['positive_rate']:.2%}。
- 采用**时间外验证**，训练集和测试集来自不同时间窗口；模型选择只使用训练集的 5 折交叉验证，测试集只在最后评估一次。
- 特征共 {model['feature_count']} 个，覆盖 R/F/M、购买间隔、近 90 天活跃度、取消行为、客户年龄等。

### 4.2 结果

| 模型 | 交叉验证 AUC | 测试 AUC | PR-AUC | Top10% 精确率 | Lift |
|---|---:|---:|---:|---:|---:|
| 逻辑回归 | {best['cv_auc_mean']:.3f} | {best['roc_auc']:.3f} | {best['pr_auc']:.3f} | {best['top_10pct']['precision']:.1%} | {best['top_10pct']['lift']:.2f} |
| 梯度提升树 | {gb['cv_auc_mean']:.3f} | {gb['roc_auc']:.3f} | {gb['pr_auc']:.3f} | {gb['top_10pct']['precision']:.1%} | {gb['top_10pct']['lift']:.2f} |
| 近期购买规则 | — | {baseline['roc_auc']:.3f} | {baseline['pr_auc']:.3f} | {baseline['top_10pct']['precision']:.1%} | {baseline['top_10pct']['lift']:.2f} |

- 最终模型为 **{model['best_model']}**，时间外测试 AUC **{best['roc_auc']:.3f}**，交叉验证 AUC **{best['cv_auc_mean']:.3f}**。
- Top 10% 高风险客户中 **{best['top_10pct']['precision']:.1%}** 实际复购，相对测试集基准复购率（{model['test']['positive_rate']:.1%}）提升 **{best['top_10pct']['lift']:.2f} 倍**。
- 相比“只按最近购买时间排序”的规则基线（AUC {baseline['roc_auc']:.3f}），模型 AUC 提升约 **{best['roc_auc']-baseline['roc_auc']:.3f}**。

![ROC 曲线](figures/roc_curve.png)

### 4.3 关键特征

置换重要性 Top 5：

1. 购买天数
2. 购买次数
3. 活跃跨度（天）
4. 平均每单行数
5. 近 90 天购买次数

![特征重要性](figures/feature_importance.png)

含义：**持续、频繁、近期活跃**的客户更可能复购；单纯“买得多”不如“买得频繁且最近还在买”。取消次数也有一定负向信号。

## 5. 业务建议

1. **守住英国、精选海外**：英国贡献 85.14% 销售额，先把英国的高价值客户运营做深；海外优先测试荷兰、爱尔兰、德国、法国。
2. **分层运营**：高价值忠诚客户做会员体系与专属服务；流失预警客户（657 人）用定向优惠召回；新客用第二单激励；沉睡客户控制触达成本。
3. **用模型排序，而不是全量触达**：按复购概率从高到低圈选 Top 10%，用短信/邮件优先触达，避免打扰低概率客户。
4. **商品策略看“订单数 + 复购”，不只看销售额**：排除一次性大单造成的虚假爆款；Top 20% 商品贡献 {kpis['top20_product_share_pct']:.2f}% 销售额，适合做重点备货与陈列。
5. **治理退货**：按商品/国家/客户拆解取消原因，优先处理高频退货商品，降低履约成本。

## 6. 局限性

- 数据来自单一英国零售商，时间范围为 2010-2011，结论不能直接外推到当前市场。
- 只有交易数据，没有成本、利润、流量、广告投放数据，无法计算 ROI 或做因果推断。
- 复购模型是历史回测；测试期复购率（61.51%）高于训练期（49.90%），存在季节性分布漂移，上线前需要重新校准概率并做 A/B 测试。
- RFM 分层基于固定快照日，客户会随时间迁移，建议按月重算。

## 7. 复现方式

```bash
# 1. 清洗：生成 sales / cancellations / adjustments 等 Parquet
python -m src.cleaning
# 2. RFM 客户分层
python -m src.rfm
# 3. 复购模型：生成模型指标、ROC、特征重要性
python -m src.repurchase
# 4. 业务分析、图表、报告
python -m src.analysis
# 5. 看板
streamlit run app/dashboard.py
```
"""
    (REPORTS_DIR / "analysis_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sales, rfm, cleaning, model = load_data()
    kpis = compute_kpis(sales, rfm, cleaning)
    (REPORTS_DIR / "kpis.json").write_text(
        json.dumps(kpis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_monthly(sales)
    plot_country(sales, kpis["uk_share_pct"])
    plot_rfm(rfm)
    plot_roc()
    plot_importance()
    write_report(sales, rfm, cleaning, model, kpis)

    print(json.dumps(kpis, ensure_ascii=False, indent=2))
    print("\n已生成图表和报告：")
    for path in sorted(FIGURES_DIR.glob("*.png")):
        print(" -", path)


if __name__ == "__main__":
    main()
