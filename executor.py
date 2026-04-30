import base64
import io
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_import(name, *args, **kwargs):
    allowed = ["pandas", "numpy", "matplotlib"]
    if name.split(".")[0] in allowed:
        return __import__(name, *args, **kwargs)


def validate_code(code: str):
    if ".boxplot()" in code and "df[" in code:
        raise ValueError(".boxplot() Series.boxplot")
    return code


def execute_python(code: str, dataframe: pd.DataFrame) -> dict:
    local_env = {"pd": pd, "np": np, "plt": plt, "df": dataframe.copy()}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    images = []

    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        plt.close("all")

        exec_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "bool": bool,
                "True": True,
                "False": False,
                "None": None,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "isinstance": isinstance,
                "zip": zip,
                "enumerate": enumerate,
                "sorted": sorted,
                "reversed": reversed,
                "__import__": safe_import,
            }
        }
        code = code.replace(".boxplot()", "")
        code = validate_code(code)

        try:
            exec(code, exec_globals, local_env)
        except Exception as e:
            print(e)

            safe_code = """
            numeric = df.select_dtypes(include=np.number)
            for col in numeric.columns:
                fig, ax = plt.subplots()
                ax.hist(numeric[col].dropna(), bins=20)
            """
            exec(safe_code, exec_globals, local_env)

        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close(fig)

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        text_output = stdout_buf.getvalue() + stderr_buf.getvalue()
        return {"text_output": text_output, "images": images, "error": None}

    except Exception as e:
        return {
            "text_output": "",
            "images": [],
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        }
