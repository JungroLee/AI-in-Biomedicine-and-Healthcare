import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def inspect_columns(df, top_n: int = 10) -> pd.DataFrame:
    """
    각 컬럼의 dtype, 유니크 개수, 결측치 개수, 샘플 값을 요약하는 함수.
    """
    rows = []
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        samples = df[col].unique()[:top_n]
        missing = df[col].isna().sum()

        rows.append({
            "column": col,
            "dtype": str(dtype),
            "nunique": nunique,
            "missing": missing,
            "sample_values": samples
        })

    return pd.DataFrame(rows)


class FeatureInspector:
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_cols=None,
        numeric_cols=None,
        target_col=None,
    ):
        """
        df: 분석 대상 DataFrame
        categorical_cols: 명시적으로 지정하지 않으면 object/category 타입 자동 탐지
        numeric_cols: 명시적으로 지정하지 않으면 수치형(np.number) 타입 자동 탐지
        target_col: 타깃 컬럼 이름(있으면 categorical/numeric 목록에서 제외)
        """
        self.df = df
        self.target_col = target_col

        # 카테고리 컬럼 자동 추론 (object, category 타입)
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        # target은 제외
        if target_col is not None and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        self.categorical_cols = categorical_cols

        # 숫자형 컬럼 자동 추론
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(
                include=[np.number]
            ).columns.tolist()
        # target은 제외
        if target_col is not None and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        self.numeric_cols = numeric_cols

    def summarize(self):
        """
        카테고리/수치형 feature의 기본 통계 요약 출력.
        """
        print("=== Categorical Features ===")
        for col in self.categorical_cols:
            print(f"\n[{col}]")
            print(self.df[col].value_counts(dropna=False))

        print("\n=== Numeric Features ===")
        for col in self.numeric_cols:
            s = self.df[col]
            print(f"\n[{col}]")
            print(f"  mean: {s.mean():.3f}")
            print(f"  std:  {s.std():.3f}")

    # ① 결측치를 각 열의 median으로 대체하는 함수
    def fill_missing_with_median(self, inplace: bool = True) -> pd.DataFrame:
        """
        numeric_cols에 대해 NaN이 있으면 각 열의 median으로 대체한다.
        categorical_cols에 NaN이 있는 경우는 대체하지 않고 개수만 출력한다.
        """
        if inplace:
            df = self.df
        else:
            df = self.df.copy()

        medians = {}

        print("=== Fill Missing (Numeric Columns, median) ===")
        for col in self.numeric_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                medians[col] = median_val
                print(f"[{col}] missing: {missing} → filled with median={median_val:.3f}")
            else:
                print(f"[{col}] no missing values")

        # 카테고리 컬럼의 결측치는 일단 보고만 하기
        print("\n=== Missing in Categorical Columns (not filled) ===")
        for col in self.categorical_cols:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"[{col}] missing: {missing} (not filled)")
            else:
                print(f"[{col}] no missing values")

        # inplace=True면 self.df가 이미 수정됨
        if inplace:
            self.df = df
        return df

    # ② 중복 행을 분석하는 함수
    def analyze_duplicates(self, subset=None) -> pd.DataFrame:
        """
        중복된 행이 있는지 확인하고, 있으면 그 행들만 모은 DataFrame을 반환한다.
        subset: 특정 컬럼 리스트로 중복 기준 지정 가능 (기본은 전체 열)
        """
        # keep=False: 중복에 해당하는 모든 row를 True로 표시
        dup_mask = self.df.duplicated(subset=subset, keep=False)
        n_dup_rows = dup_mask.sum()

        if subset is None:
            subset_info = "all columns"
        else:
            subset_info = f"columns={subset}"

        print("=== Duplicate Row Analysis ===")
        print(f"Criteria: {subset_info}")
        print(f"Number of duplicated rows: {n_dup_rows}")

        dup_df = self.df[dup_mask]

        if n_dup_rows > 0:
            print("Duplicated rows found. Returning duplicated subset.")
        else:
            print("No duplicated rows found.")

        return dup_df

    def has_duplicates(self, subset=None) -> bool:
        """
        중복 존재 여부만 True/False로 반환.
        """
        dup_exists = self.df.duplicated(subset=subset).any()

        if subset is None:
            subset_info = "all columns"
        else:
            subset_info = f"columns={subset}"

        print(f"=== Duplicate Existence Check ({subset_info}) ===")
        print(f"Duplicated rows exist? {dup_exists}")

        return dup_exists

    def plot_categorical_counts(self):
        """
        모든 categorical_cols에 대해 value_counts barplot을
        하나의 figure 안에 subplot으로 그린다.
        """
        if len(self.categorical_cols) == 0:
            print("categorical_cols가 비어 있습니다.")
            return

        n = len(self.categorical_cols)
        ncols = 2
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        axes = np.array(axes).reshape(-1)  # 1D로 평탄화

        for ax, col in zip(axes, self.categorical_cols):
            vc = self.df[col].value_counts(dropna=False)
            ax.bar(vc.index.astype(str), vc.values)
            ax.set_title(col)
            ax.set_ylabel("Count")
            ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")

        # 남는 subplot이 있으면 비우기
        for ax in axes[len(self.categorical_cols):]:
            ax.axis("off")

        fig.suptitle("Categorical Feature Distributions", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_corr_heatmap(
        self,
        method: str = "pearson",
        include_categorical: bool = False,
        annotate: bool = True,
    ):
        """
        상관관계 heatmap을 그린다.
        - 기본: numeric_cols만 사용
        - include_categorical=True 이면 categorical_cols도 임시 정수 인코딩해서 포함
        - annotate=True 이면 각 셀에 상관계수 숫자를 표시
        """
        # numeric 기반 DataFrame
        df_for_corr = self.df[self.numeric_cols].copy()

        # categorical도 포함하고 싶을 때
        if include_categorical and len(self.categorical_cols) > 0:
            for col in self.categorical_cols:
                # factorize: 카테고리 → 0,1,2,... 정수 코드
                codes, uniques = pd.factorize(self.df[col])
                df_for_corr[col] = codes
        elif not include_categorical and len(self.numeric_cols) == 0:
            raise ValueError("numeric_cols가 비어 있습니다.")

        cols = df_for_corr.columns.tolist()
        corr = df_for_corr.corr(method=method)

        plt.figure(figsize=(1.0 * len(cols) + 4, 1.0 * len(cols) + 3))
        im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.xticks(
            ticks=np.arange(len(cols)),
            labels=cols,
            rotation=45,
            ha="right"
        )
        plt.yticks(
            ticks=np.arange(len(cols)),
            labels=cols
        )

        plt.title(f"Correlation Heatmap ({method})", fontsize=14)

        # 각 셀에 숫자 찍기
        if annotate:
            for i in range(len(cols)):
                for j in range(len(cols)):
                    val = corr.iloc[i, j]
                    if pd.isna(val):
                        text = ""
                    else:
                        text = f"{val:.2f}"
                    # 값 크기에 따라 글자색 조절(가독성)
                    color = "black"
                    if abs(val) > 0.6:
                        color = "white"
                    plt.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )

        plt.tight_layout()
        plt.show()
