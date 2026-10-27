# 电商零售数据分析（Online Retail）

基于英国在线零售商 2010-12 至 2011-12 的真实交易数据，完成数据清洗、业务指标分析、RFM 用户分层和复购预测，并用 Streamlit 交付一个可交互看板。

这是求职作品集项目，目标是完整走通「数据 → 分析 → 建模 → 展示」流程，并记录每一步的决策与结论。

## 数据集

- 来源：[UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- 规模：541,909 行，8 个字段
- 时间范围：2010-12-01 至 2011-12-09

| 字段 | 含义 |
|---|---|
| `InvoiceNo` | 订单号，以 `C` 开头表示取消/退货 |
| `StockCode` | 商品编码 |
| `Description` | 商品名称 |
| `Quantity` | 数量，负数表示退货 |
| `InvoiceDate` | 交易时间 |
| `UnitPrice` | 单价 |
| `CustomerID` | 客户编号，缺失表示匿名交易 |
| `Country` | 国家/地区 |

原始数据不提交到 Git。下载后放到 `data/raw/Online Retail.xlsx`。

## 项目结构

```
retail-analysis/
├── data/
│   ├── raw/              # 原始数据（不入库）
│   └── processed/        # 清洗后的数据
├── notebooks/            # 分析过程
├── src/                  # 可复用脚本
├── reports/              # 图表、分析报告
├── assets/               # README 图片等素材
├── requirements.txt
└── README.md
```

## 环境

- Python 3.12
- 依赖见 `requirements.txt`

```bash
# 1. 创建虚拟环境
py -3.12 -m venv .venv

# 2. 激活（Git Bash）
source .venv/Scripts/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 注册 Jupyter 内核（只需执行一次）
python -m ipykernel install --user --name ds-python312 --display-name "Python 3.12 (retail-analysis)"
```

PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。

启动 Jupyter：

```bash
jupyter notebook
```

打开 Notebook 时选择 `Python 3.12 (retail-analysis)` 内核。

## 分析计划

- [x] 环境搭建与数据下载
- [x] 数据加载与初探
- [ ] 数据清洗（缺失值、退货、异常价格）
- [ ] 业务指标分析（趋势、国家、商品、客户）
- [ ] RFM 用户分层
- [ ] 复购/流失预测
- [ ] Streamlit 看板
- [ ] 分析报告与项目总结

## 指标口径

| 指标 | 定义 |
|---|---|
| 销售额 | `Quantity × UnitPrice` |
| 有效销售 | 排除退货（`Quantity <= 0`）和异常价格（`UnitPrice <= 0`） |
| 订单数 | 去重 `InvoiceNo` |
| 客户数 | 去重 `CustomerID`（排除缺失） |
| 复购客户 | 统计期内下单次数 ≥ 2 的客户 |

## 当前进度

- [x] 创建项目与环境
- [x] 下载数据
- [x] Notebook 01：数据加载与初探

## 说明

- 本项目仅用于学习与求职展示，数据版权归原始数据集作者所有。
- 所有分析结论均基于公开数据，不代表真实企业的经营情况。
