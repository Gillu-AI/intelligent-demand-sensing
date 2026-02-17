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

        self.date_col = config["data_schema"]["sales"]["date_column"]
        self.target_col = config["data_schema"]["sales"]["target_column"]

        prophet_config = config["modeling"]["models"]["prophet"]

        self.model = Prophet(
            yearly_seasonality=prophet_config.get("yearly_seasonality", True),
            weekly_seasonality=prophet_config.get("weekly_seasonality", True)
        )

    def fit(self, df: pd.DataFrame):
        """
        Fit Prophet model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing date and target columns.
        """

        prophet_df = df[[self.date_col, self.target_col]].copy()
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

        future_df = df[[self.date_col]].copy()
        future_df.columns = ["ds"]

        forecast = self.model.predict(future_df)

        return forecast["yhat"].values
