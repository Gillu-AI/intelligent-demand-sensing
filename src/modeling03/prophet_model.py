# src/modeling03/prophet_model.py

from typing import Dict
import pandas as pd
from prophet import Prophet


class ProphetRegressor:
    """
    Wrapper class for Facebook Prophet to integrate with IDS architecture.

    Notes
    -----
    - Uses only date and target column.
    - Ignores engineered features.
    - Compatible with evaluate.py output.
    - Designed to behave similar to sklearn model.
    """

    def __init__(self, config: Dict):
        """
        Initialize Prophet model using configuration.

        Parameters
        ----------
        config : Dict
            Full configuration dictionary.
        """

        if "data_schema" not in config or "sales" not in config["data_schema"]:
            raise ValueError("Missing 'data_schema.sales' configuration.")

        if (
            "modeling" not in config
            or "models" not in config["modeling"]
            or "prophet" not in config["modeling"]["models"]
        ):
            raise ValueError("Missing 'modeling.models.prophet' configuration.")

        self.date_col = config["data_schema"]["sales"]["date_column"]
        self.target_col = config["data_schema"]["sales"]["target_column"]

        prophet_config = config["modeling"]["models"]["prophet"]

        self.model = Prophet(
            yearly_seasonality=prophet_config.get("yearly_seasonality", True),
            weekly_seasonality=prophet_config.get("weekly_seasonality", True),
        )

    def fit(self, df: pd.DataFrame):
        """
        Fit Prophet model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing date and target columns.
        """

        if self.date_col not in df.columns:
            raise ValueError(f"Column '{self.date_col}' not found in dataset.")

        if self.target_col not in df.columns:
            raise ValueError(f"Column '{self.target_col}' not found in dataset.")

        prophet_df = df[[self.date_col, self.target_col]].copy()

        if not pd.api.types.is_datetime64_any_dtype(prophet_df[self.date_col]):
            prophet_df[self.date_col] = pd.to_datetime(
                prophet_df[self.date_col], errors="raise"
            )

        prophet_df.columns = ["ds", "y"]

        self.model.fit(prophet_df)

        return self

    def predict(self, df: pd.DataFrame):
        """
        Generate predictions.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing date column.

        Returns
        -------
        pd.Series
            Predicted values aligned with input index.
        """

        if self.date_col not in df.columns:
            raise ValueError(f"Column '{self.date_col}' not found in dataset.")

        future_df = df[[self.date_col]].copy()

        if not pd.api.types.is_datetime64_any_dtype(future_df[self.date_col]):
            future_df[self.date_col] = pd.to_datetime(
                future_df[self.date_col], errors="raise"
            )

        future_df.columns = ["ds"]

        forecast = self.model.predict(future_df)

        return pd.Series(
            forecast["yhat"].values,
            index=df.index,
            name="prophet_prediction",
        )
