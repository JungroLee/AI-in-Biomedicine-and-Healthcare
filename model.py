# model.py
import numpy as np
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.optim as optim


# ============================
# PyTorch 기반 MLP 분류기
# ============================

class TorchMLPClassifier:
    """
    PyTorch로 구현한 MLP 이진 분류기.
    - fairness_lambda=0.0 이면 일반 BCE 로스만 사용
    - fairness_lambda>0.0 이면 BCE + fairness penalty 사용

      fairness_type:
        - "dp"  : Demographic Parity 스타일
                  (batch 내에서 group=0,1 의 전체 평균 확률 차이 제곱)
        - "tpr" : Equal Opportunity (soft TPR) 스타일
                  (batch 내에서 y=1 인 샘플만 골라 group=0,1 평균 확률 차이 제곱)

    sklearn 인터페이스 비슷하게:
      - fit(X, y, group=None)
      - predict_proba(X)
      - predict(X)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(64, 32),
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        fairness_lambda: float = 0.0,
        fairness_type: str = "dp",      # "dp" or "tpr"
        random_state: int = 42,
        device: Optional[str] = None,
        name: str = "mlp",
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.fairness_lambda = fairness_lambda
        self.fairness_type = fairness_type
        self.random_state = random_state
        self.name = name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 네트워크 정의
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # binary output (logit)
        self.model = nn.Sequential(*layers).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self._is_fitted = False
        self.group_mapping_ = None  # 원래 group 라벨 -> int 코드 매핑 (optional)

    def fit(self, X, y, group=None):
        """
        X: (n_samples, n_features) numpy or pandas
        y: (n_samples,) 0/1
        group: (n_samples,) 그룹 레이블 (예: 0/1, 'M'/'F')
               fairness_lambda>0일 때 반드시 필요
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        if self.fairness_lambda > 0.0:
            if group is None:
                raise ValueError("fairness_lambda>0 인 경우 group 벡터가 필요합니다.")
            group = np.asarray(group)

            # 문자열/카테고리일 경우 정수 인코딩 (0,1,...)
            if group.dtype.kind in ("U", "S", "O"):  # unicode, string, object
                unique, inv = np.unique(group, return_inverse=True)
                # 예: {'F':0, 'M':1} 같은 매핑
                self.group_mapping_ = {val: int(idx) for idx, val in enumerate(unique)}
                group = inv.astype(np.int64)
            else:
                group = group.astype(np.int64)
        else:
            group = None

        dataset_size = X.shape[0]
        indices = np.arange(dataset_size)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            group_shuffled = group[indices] if group is not None else None

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                xb_t = torch.tensor(xb, device=self.device)
                yb_t = torch.tensor(yb, device=self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb_t)
                bce_loss = self.criterion(logits, yb_t)

                loss = bce_loss

                # 공정성 페널티
                if self.fairness_lambda > 0.0:
                    gb = group_shuffled[start:end]
                    gb_t = torch.tensor(gb, dtype=torch.long, device=self.device)
                    probs = torch.sigmoid(logits)  # (batch, 1)

                    if self.fairness_type == "dp":
                        # Demographic Parity: 전체 평균 확률 차이
                        mask0 = gb_t == 0
                        mask1 = gb_t == 1

                        if mask0.any() and mask1.any():
                            pos_rate0 = probs[mask0].mean()
                            pos_rate1 = probs[mask1].mean()
                            fair_penalty = (pos_rate0 - pos_rate1) ** 2
                        else:
                            fair_penalty = torch.tensor(0.0, device=self.device)

                    elif self.fairness_type == "tpr":
                        # Equal Opportunity (soft TPR): y=1 인 샘플에서만 평균 확률 차이
                        mask_pos = (yb_t == 1.0).view(-1)  # True for positive label
                        mask0 = (gb_t == 0) & mask_pos
                        mask1 = (gb_t == 1) & mask_pos

                        if mask0.any() and mask1.any():
                            tpr0 = probs.view(-1)[mask0].mean()
                            tpr1 = probs.view(-1)[mask1].mean()
                            fair_penalty = (tpr0 - tpr1) ** 2
                        else:
                            fair_penalty = torch.tensor(0.0, device=self.device)

                    else:
                        raise ValueError(f"Unknown fairness_type: {self.fairness_type}")

                    loss = bce_loss + self.fairness_lambda * fair_penalty

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # 필요하면 로그 출력
            # print(f"[{self.name}] Epoch {epoch+1}/{self.epochs}, loss={epoch_loss/n_batches:.4f}")

        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        xb_t = torch.tensor(X, device=self.device)
        with torch.no_grad():
            logits = self.model(xb_t)
            probs1 = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs0 = 1.0 - probs1
        return np.vstack([probs0, probs1]).T  # (n_samples, 2)

    def predict(self, X, threshold: float = 0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ============================
# 공통 metric 함수들
# ============================

def _binary_metrics(y_test, y_score, threshold: float = 0.5):
    """
    Internal helper: compute binary classification metrics from scores.
    """
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================
# 기존 ML 모델 + MLP(BCE)
# ============================

def get_models(input_dim: int, random_state: int = 42) -> Dict[str, object]:
    """
    Return dict of models to be evaluated.
    Linear/Ridge are regression models but treated as binary classifiers
    by thresholding their continuous outputs at 0.5.

    input_dim: number of features (for MLP)
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=random_state),
        # MLP (BCE only, fairness_lambda=0)
        "mlp_bce": TorchMLPClassifier(
            input_dim=input_dim,
            hidden_dims=(64, 32),
            lr=1e-3,
            epochs=100,
            batch_size=64,
            fairness_lambda=0.0,
            fairness_type="dp",   # 의미 없음 (lambda=0)
            random_state=random_state,
            name="mlp_bce",
        ),
    }
    return models


def evaluate_models(
    models: Dict[str, object],
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Fit each model and evaluate on test set.

    For models without predict_proba (e.g., Linear/Ridge regression),
    the continuous output is thresholded at `threshold` to obtain binary predictions.

    Returns:
        DataFrame with per-model metrics:
        accuracy, precision, recall, f1, roc_auc, confusion_matrix, tn, fp, fn, tp
    """
    rows = []

    for name, model in models.items():
        # fit
        if isinstance(model, TorchMLPClassifier):
            # MLP는 y만으로 학습 (fairness 없는 버전)
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # score / prob
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict(X_test).ravel()

        metrics = _binary_metrics(y_test, y_score, threshold=threshold)

        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df


# ============================
# 모델 비교 plot
# ============================

def plot_model_metrics(
    results_df: pd.DataFrame,
    metrics: Iterable[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
):
    """
    Plot bar charts comparing multiple models across several metrics
    in a single figure (one subplot per metric).
    """
    models = results_df["model"].tolist()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=False)

    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(models))

    for ax, metric in zip(axes, metrics):
        if metric not in results_df.columns:
            ax.set_visible(False)
            continue

        values = results_df[metric].values
        ax.bar(x, values)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Model Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================
# 그룹별 성능 (fairness 분석)
# ============================

def evaluate_models_by_group(
    models: Dict[str, object],
    X_test,
    y_test,
    group_values,
    threshold: float = 0.5,
    group_name: str = "group",
) -> pd.DataFrame:
    """
    Using already-fitted models, evaluate performance per group
    (e.g., by Sex: M/F).
    """
    group_values = np.asarray(group_values)
    unique_groups = np.unique(group_values)

    rows = []

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score_all = model.predict_proba(X_test)[:, 1]
        else:
            y_score_all = model.predict(X_test).ravel()

        for g in unique_groups:
            mask = group_values == g
            if mask.sum() == 0:
                continue

            y_test_g = y_test[mask]
            y_score_g = y_score_all[mask]

            metrics = _binary_metrics(y_test_g, y_score_g, threshold=threshold)

            row = {
                "model": name,
                group_name: g,
            }
            row.update(metrics)
            rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df


def plot_fairness_metric(
    group_results_df: pd.DataFrame,
    metric: str = "recall",
    group_name: str = "Sex",
):
    """
    For a single metric (e.g. recall), plot group-wise (e.g. M/F) bars
    per model.
    """
    if metric not in group_results_df.columns:
        raise ValueError(f"{metric} not found in group_results_df.columns")

    pivot = group_results_df.pivot(
        index="model",
        columns=group_name,
        values=metric,
    )

    groups = pivot.columns.tolist()
    models = pivot.index.tolist()

    x = np.arange(len(models))
    width = 0.8 / max(len(groups), 1)

    plt.figure(figsize=(6 + 2 * len(groups), 4))

    for i, g in enumerate(groups):
        vals = pivot[g].values
        plt.bar(x + i * width, vals, width=width, label=str(g))

    plt.xticks(x + width * (len(groups) - 1) / 2, models, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by {group_name}")
    plt.ylim(0, 1.0)
    plt.legend(title=group_name)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fairness_all_metrics(
    group_results_df: pd.DataFrame,
    metrics: Iterable[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
    group_name: str = "Sex",
):
    """
    Plot fairness (group-wise performance) for multiple metrics in one figure.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=False)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in group_results_df.columns:
            ax.set_visible(False)
            continue

        pivot = group_results_df.pivot(
            index="model",
            columns=group_name,
            values=metric,
        )

        groups = pivot.columns.tolist()
        models = pivot.index.tolist()
        x = np.arange(len(models))
        width = 0.8 / max(len(groups), 1)

        for i, g in enumerate(groups):
            vals = pivot[g].values
            ax.bar(x + i * width, vals, width=width, label=str(g))

        ax.set_title(metric)
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title=group_name, loc="upper right")

    fig.suptitle(f"Fairness by {group_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================
# Fairness MLP (BCE + penalty)만 따로 평가하고 싶을 때
# ============================

def evaluate_fair_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    group_train,
    fairness_lambda: float,
    fairness_type: str = "tpr",   # 기본을 soft TPR penalty로
    hidden_dims=(64, 32),
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    BCE + fairness penalty를 사용하는 MLP 하나를 따로 학습/평가해서
    다른 모델들과 비교 가능한 형태(pd.DataFrame 한 행)로 반환.

    fairness_type:
        - "dp"  : 전체 평균 확률 차이 (Demographic Parity 스타일)
        - "tpr" : y=1 에서의 평균 확률 차이 (Equal Opportunity 스타일)
    """
    input_dim = X_train.shape[1]
    model_name = f"mlp_bce_{fairness_type}_{fairness_lambda}"

    mlp = TorchMLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        fairness_lambda=fairness_lambda,
        fairness_type=fairness_type,
        random_state=random_state,
        name=model_name,
    )

    mlp.fit(X_train, y_train, group=group_train)

    y_score = mlp.predict_proba(X_test)[:, 1]
    metrics = _binary_metrics(y_test, y_score, threshold=0.5)

    row = {"model": model_name}
    row.update(metrics)

    return pd.DataFrame([row])
    
def plot_fairness_gap(
    group_results_df: pd.DataFrame,
    metrics: Iterable[str] = ("accuracy", "precision", "recall", "f1", "roc_auc"),
    group_name: str = "Sex",
):
    """
    각 metric에 대해, group 간(예: Male vs Female) 절댓값 차이(|metric_F - metric_M|)
    를 모델별로 bar plot으로 그려준다.

    group_results_df 는 evaluate_models_by_group 의 결과 DataFrame 이고,
    반드시 한 metric에 대해 group_name 컬럼이 정확히 2개의 group 값을 가져야 한다
    (예: 'F', 'M').
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=False)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in group_results_df.columns:
            ax.set_visible(False)
            continue

        # model x group pivot
        pivot = group_results_df.pivot(
            index="model",
            columns=group_name,
            values=metric,
        )

        groups = pivot.columns.tolist()
        if len(groups) != 2:
            raise ValueError(
                f"{group_name} 에 대해 정확히 2개 group이 필요합니다. 현재: {groups}"
            )

        g1, g2 = groups[0], groups[1]
        gap = (pivot[g1] - pivot[g2]).abs()   # 절댓값 차이

        models = pivot.index.tolist()
        x = np.arange(len(models))

        ax.bar(x, gap.values)
        ax.set_title(f"|{metric}({g1}) - {metric}({g2})|")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{group_name} gap per model", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
