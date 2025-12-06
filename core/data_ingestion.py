# data_ingestion.py

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import io


def load_csv_file(uploaded_file) -> Dict[str, Any]:
    """
    Load a CSV-like file from Streamlit uploader and return df + simple metadata.
    Supports CSV and Excel.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    filename = uploaded_file.name.lower()

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        # Try robust CSV parsing
        content = uploaded_file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep=";")

    df = df.dropna(how="all").reset_index(drop=True)

    meta = {
        "file_name": uploaded_file.name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    return {"df": df, "meta": meta}
