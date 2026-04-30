import numpy as np
import pandas as pd


def compress_dataframe(df: pd.DataFrame) -> dict:
    summary = {}

    summary["shape"] = df.shape

    summary["columns"] = {}
    for col in df.columns:
        col_data = df[col]

        col_summary = {
            "dtype": str(col_data.dtype),
            "missing": int(col_data.isnull().sum()),
            "unique": int(col_data.nunique()),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            col_summary.update(
                {
                    "mean": (
                        float(col_data.mean())
                        if not col_data.isnull().all()
                        else None
                    ),
                    "std": (
                        float(col_data.std())
                        if not col_data.isnull().all()
                        else None
                    ),
                    "min": (
                        float(col_data.min())
                        if not col_data.isnull().all()
                        else None
                    ),
                    "max": (
                        float(col_data.max())
                        if not col_data.isnull().all()
                        else None
                    ),
                }
            )
        else:
            top_values = col_data.value_counts().head(3)
            col_summary["top_values"] = top_values.to_dict()

        summary["columns"][col] = col_summary

    if len(df.select_dtypes(include=np.number).columns) > 1:
        corr = df.select_dtypes(include=np.number).corr().round(2)
        summary["correlations"] = corr.to_dict()

    return summary 
