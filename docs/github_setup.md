# GitHub 仓库整理清单

## 1. 仓库名称

建议使用：

- `retail-analysis`（短、清晰，推荐）
- `ecommerce-retail-analysis`（更具体）

## 2. About 描述（直接粘贴到仓库右上角 About）

中文：

> 54 万条真实交易数据的数据分析项目：清洗、RFM 客户分层、复购预测与 Streamlit 看板。

English：

> End-to-end retail analytics: data cleaning, RFM segmentation, repurchase prediction, and an interactive Streamlit dashboard.

## 3. Topics（直接粘贴，逗号分隔）

```text
data-analysis, pandas, scikit-learn, rfm, customer-segmentation, repurchase-prediction, streamlit, python, portfolio-project
```

## 4. 推送到 GitHub

先在 GitHub 新建一个空仓库（不要勾选自动生成 README），然后在项目目录执行：

```bash
cd D:/ds-portfolio/retail-analysis
git remote add origin https://github.com/<你的用户名>/retail-analysis.git
git branch -M main
git push -u origin main
```

如果已经配置过 remote，改地址：

```bash
git remote set-url origin https://github.com/<你的用户名>/retail-analysis.git
git push -u origin main
```

认证方式：

- 推荐安装 GitHub CLI，执行 `gh auth login` 后按提示登录。
- 或使用 Personal Access Token（PAT）作为密码；不要把账号密码直接贴在命令里。
- 如果推送被拒绝，先确认仓库是空的，且你有写权限。

推送后检查：

1. 打开仓库主页，确认 README 的图片能正常显示。
2. 确认 `data/raw`、`.venv`、`data/processed` 没有被提交。
3. 可选：Settings → Social preview，上传 `reports/figures/monthly_revenue.png` 作为仓库预览图。

## 5. 置顶仓库

1. 打开 GitHub 个人主页。
2. 点击 **Customize your pins**。
3. 勾选 `retail-analysis`，保存。
4. 简历项目标题后加：

```text
项目代码：github.com/<你的用户名>/retail-analysis
```

如果仓库是私有的，不要放一个打不开的链接：要么改成 public，要么写“代码可提供”。

## 6. 可选：GitHub 个人主页 README

新建一个和用户名同名的仓库（例如用户名是 `Keys`，仓库名就是 `Keys`），在根目录放 `README.md`：

```markdown
### Hi, I'm Keys

Data Science / Big Data student, focused on data analysis, customer analytics and machine learning.

- Featured project: [retail-analysis](https://github.com/<你的用户名>/retail-analysis)
- Tools: Python, pandas, SQL, scikit-learn, Streamlit, Git
- Currently learning: Spark / Flink and production data pipelines

> 把 `<你的用户名>` 替换成真实用户名；邮箱等隐私信息按需填写。
```

## 7. 最后检查

```bash
cd D:/ds-portfolio/retail-analysis
git status
git log --oneline -3
git ls-files | grep -iE "raw|venv|parquet"   # Git Bash
```

最后一条**没有输出是正常的**；如果出现 `data/raw`、`.venv` 或原始 Excel，说明它们被误提交了，需要从 Git 中移除并检查 `.gitignore`。
