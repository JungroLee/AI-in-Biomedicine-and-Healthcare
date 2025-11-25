import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


class DataPreprocessor:
    def __init__(
        self,
        categorical_cols: List[str],
        numeric_cols: List[str],
        target_col: Optional[str] = None,
    ):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.target_col = target_col

        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        # 각 카테고리 컬럼마다 LabelEncoder 학습
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col])
            self.encoders[col] = le

        # 숫자 컬럼에 대해 StandardScaler 학습
        self.scaler.fit(df[self.numeric_cols])

        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")

        out = df.copy()

        # 카테고리 → 정수 인코딩
        for col, le in self.encoders.items():
            out[col] = le.transform(out[col])

        # 숫자 → 표준화
        out[self.numeric_cols] = self.scaler.transform(out[self.numeric_cols])

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    # ============================
    # ① numeric feature boxplot
    # ============================
    def plot_boxplots(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
    ) -> None:
        """
        numeric_cols에 대해 boxplot을 그려 outlier 분포를 시각적으로 확인.
        cols를 지정하면 그 컬럼들만 그림.
        """
        if cols is None:
            cols = self.numeric_cols

        for col in cols:
            plt.figure()
            plt.boxplot(df[col].dropna(), vert=True)
            plt.title(f"Boxplot of {col}")
            plt.ylabel(col)
            plt.grid(True, axis="y", alpha=0.3)
            plt.show()

    # ============================
    # ② IQR 기반 outlier clipping
    # ============================
    def clip_outliers_iqr(
        self,
        df: pd.DataFrame,
        cols: Optional[List[str]] = None,
        factor: float = 1.5,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        IQR 기반으로 outlier를 [Q1 - factor*IQR, Q3 + factor*IQR] 범위로 clip.

        - df: 대상 DataFrame (보통 train_df)
        - cols: 처리할 numeric 컬럼 리스트 (None이면 self.numeric_cols 사용)
        - factor: IQR 배수 (기본 1.5, 더 강하게 자르고 싶으면 3.0 등)
        - inplace: True면 df를 직접 수정, False면 복사본 반환
        """
        if cols is None:
            cols = self.numeric_cols

        if inplace:
            out = df
        else:
            out = df.copy()

        print("=== IQR-based Outlier Clipping ===")
        for col in cols:
            s = out[col].dropna()
            if s.empty:
                print(f"[{col}] skipped (all values are NaN)")
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                print(f"[{col}] skipped (IQR = 0)")
                continue

            lower = q1 - factor * iqr
            upper = q3 + factor * iqr

            before_outliers = ((out[col] < lower) | (out[col] > upper)).sum()
            out[col] = out[col].clip(lower, upper)

            print(
                f"[{col}] clipped {before_outliers} values "
                f"to range [{lower:.3f}, {upper:.3f}]"
            )

        return out
    def plot_all_boxplots(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        df[self.numeric_cols].boxplot()
        plt.title("Boxplot of All Numeric Features")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.show()
