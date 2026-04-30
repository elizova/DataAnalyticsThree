import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fallback_plots(df):
    images = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return images

    for col in numeric_cols[:3]:
        try:
            fig, ax = plt.subplots()
            ax.hist(
                df[col].dropna(), bins=20, color="skyblue", edgecolor="black"
            )
            ax.set_title(f"Распределение: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Частота")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close(fig)
        except:
            pass

    if len(numeric_cols) > 1:
        try:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.matshow(corr, cmap="coolwarm")
            fig.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            ax.set_title("Матрица корреляций")
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(
                        j,
                        i,
                        f"{corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                    )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close(fig)
        except:
            pass
    return images
