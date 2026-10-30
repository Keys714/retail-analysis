"""复购预测模型：时间外验证、模型对比、指标输出。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from src.features import FEATURE_COLUMNS, build_train_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def top_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> dict:
    """按预测概率从高到低取前 k 比例，计算召回率、精确率和提升倍数。"""
    n = max(int(len(y_true) * k), 1)
    order = np.argsort(-y_score)
    selected = y_true[order[:n]]
    positives = y_true.sum()

    precision = selected.sum() / n
    recall = selected.sum() / positives if positives else 0.0
    base_rate = positives / len(y_true)
    lift = precision / base_rate if base_rate else 0.0
    return {
        "k_pct": k * 100,
        "n_selected": int(n),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "lift": round(float(lift), 4),
    }


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "top_10pct": top_k_metrics(y_true, y_score, 0.10),
        "top_20pct": top_k_metrics(y_true, y_score, 0.20),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train, test = build_train_test()
    X_train, y_train = train[FEATURE_COLUMNS], train["label"].to_numpy()
    X_test, y_test = test[FEATURE_COLUMNS], test["label"].to_numpy()

    models = {
        "逻辑回归": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "scaler",
                    QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=min(1000, len(train)),
                        random_state=42,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "梯度提升树": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    predictions = {}
    for name, model in models.items():
        # 只用训练集做 5 折交叉验证来选模型，测试集只在最后评估一次
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        predictions[name] = prob
        result = evaluate(name, y_test, prob)
        result["cv_auc_mean"] = round(float(cv_scores.mean()), 4)
        result["cv_auc_std"] = round(float(cv_scores.std()), 4)
        results.append(result)

    # 基线 1：全部预测为少数类/多数类中的较优者
    majority_rate = max(y_test.mean(), 1 - y_test.mean())
    # 基线 2：只按最近一次购买时间排序（越近越可能复购）
    recency_score = -test["recency_days"].to_numpy()
    results.append(
        {
            "model": "基线：近期购买规则",
            "roc_auc": round(float(roc_auc_score(y_test, recency_score)), 4),
            "pr_auc": round(float(average_precision_score(y_test, recency_score)), 4),
            "top_10pct": top_k_metrics(y_test, recency_score, 0.10),
            "top_20pct": top_k_metrics(y_test, recency_score, 0.20),
        }
    )
    results.append(
        {
            "model": "基线：多数类",
            "accuracy": round(float(majority_rate), 4),
            "positive_rate_test": round(float(y_test.mean()), 4),
        }
    )

    best_name = max(
        [r for r in results if "cv_auc_mean" in r],
        key=lambda r: r["cv_auc_mean"],
    )["model"]
    best_score = predictions[best_name]
    best_model = models[best_name]

    # 特征重要性（在测试集上做置换重要性，衡量对 AUC 的贡献）
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # ROC 曲线数据
    fpr, tpr, _ = roc_curve(y_test, best_score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        REPORTS_DIR / "roc_curve.csv", index=False
    )

    # 保存测试集预测，供看板展示
    test_out = test[
        [
            "customer_id",
            "snapshot",
            "recency_days",
            "frequency",
            "monetary",
            "label",
        ]
    ].copy()
    test_out["probability"] = best_score
    test_out["predicted"] = (best_score >= 0.5).astype(int)
    test_out.sort_values("probability", ascending=False).to_parquet(
        PROCESSED_DIR / "repurchase_test_scores.parquet", index=False
    )

    joblib.dump(best_model, MODELS_DIR / "repurchase_model.joblib")

    metrics = {
        "train": {
            "snapshot": str(train["snapshot"].iloc[0]),
            "customers": int(len(train)),
            "positive_rate": round(float(y_train.mean()), 4),
        },
        "test": {
            "snapshot": str(test["snapshot"].iloc[0]),
            "customers": int(len(test)),
            "positive_rate": round(float(y_test.mean()), 4),
        },
        "feature_count": len(FEATURE_COLUMNS),
        "results": results,
        "best_model": best_name,
        "feature_importance_top10": importance_df.head(10).to_dict(
            orient="records"
        ),
    }
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
