import numpy as np
import pandas as pd


def engineer_features(transaction_df):
    """
    Apply the same feature engineering used during model training.
    """

    df_input = transaction_df.copy()

    # Time-based features
    df_input["Hour"] = (
        df_input["Time"] // 3600
    ) % 24

    df_input["Day"] = (
        df_input["Time"] // 86400
    ).astype(int)

    # Cyclical encoding of hour
    df_input["Hour_sin"] = np.sin(
        2 * np.pi * df_input["Hour"] / 24
    )

    df_input["Hour_cos"] = np.cos(
        2 * np.pi * df_input["Hour"] / 24
    )

    # Log transformation of transaction amount
    df_input["Amount_log"] = np.log1p(
        df_input["Amount"]
    )

    return df_input