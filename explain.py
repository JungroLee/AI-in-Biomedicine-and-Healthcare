# model_shap.py

import numpy as np
import shap

class FairMLPShapExplainer:
    def __init__(
        self,
        model,
        X_train,
        feature_names=None,
        background_size=50,
        random_state=42,
    ):
        """
        model: TorchMLPClassifier (already fitted)
        X_train: pandas DataFrame or numpy array
        feature_names: optional list of feature names
        background_size: number of samples for KernelExplainer background
        """

        self.model = model
        self.feature_names = (
            feature_names if feature_names is not None else np.arange(X_train.shape[1])
        )

        # background sample
        if hasattr(X_train, "sample"):  # pandas DataFrame
            background = X_train.sample(background_size, random_state=random_state).values
        else:  # numpy
            idx = np.random.RandomState(random_state).choice(
                X_train.shape[0], background_size, replace=False
            )
            background = X_train[idx]

        self.background = background

        # Prediction function (SHAP expects 1D prob outputs)
        def predict_func(X):
            # model.predict_proba returns (n,2), so take class 1
            return model.predict_proba(X)[:, 1]

        self.predict_func = predict_func

        # Create KernelExplainer
        self.explainer = shap.KernelExplainer(
            predict_func,
            background
        )

        self.shap_values = None

    def compute_shap(self, X_test, nsamples=200):
        """
        Compute SHAP values for the MLP fairness model.

        X_test: dataframe or numpy
        nsamples: number of SHAP samples (higher = slower but more accurate)
        """
        if hasattr(X_test, "values"):  # pandas DataFrame
            X_np = X_test.values
        else:
            X_np = X_test

        shap_vals = self.explainer.shap_values(X_np, nsamples=nsamples)
        self.shap_values = shap_vals
        return shap_vals

    def plot_summary(self, X_test):
        """Standard SHAP summary plot."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap() first.")

        if hasattr(X_test, "values"):  # pandas
            X_np = X_test.values
        else:
            X_np = X_test

        shap.summary_plot(self.shap_values, X_np, feature_names=self.feature_names)

    def plot_bar(self, X_test):
        """Feature importance bar plot."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap() first.")

        if hasattr(X_test, "values"):  # pandas
            X_np = X_test.values
        else:
            X_np = X_test

        shap.summary_plot(self.shap_values, X_np, feature_names=self.feature_names, plot_type="bar")

    def plot_individual(self, X_row, index=0):
        """Force plot for single prediction."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap() first.")

        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[index],
            X_row,
            feature_names=self.feature_names,
            matplotlib=True
        )
