"""FastAPI service exposing fatigue pipeline processing and training endpoints."""

from __future__ import annotations

import io
from typing import Any, Dict, Tuple

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from fatigue_pipeline import (
    augment_with_high_fatigue,
    build_and_train_baselines,
    run_pipeline,
)

app = FastAPI(
    title="Fatigue Prediction API",
    version="1.0.0",
    description="Process fatigue signals and train baseline classifiers.",
)


def _read_csv_upload(upload: UploadFile) -> pd.DataFrame:
    try:
        raw = upload.file.read()
        text_stream = io.StringIO(raw.decode("utf-8"))
        df = pd.read_csv(text_stream)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=f"CSV 解析失敗: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV 內容為空")
    return df


@app.get("/healthz")
def health_check() -> Dict[str, str]:
    """Render-level health probe."""
    return {"status": "ok"}


@app.post("/process")
def process_file(
    file: UploadFile = File(...),
    augment_high: bool = False,
) -> Dict[str, Any]:
    """
    上傳 CSV，執行疲勞管線並回傳統計摘要。

    - `augment_high`: 是否自動加入高疲勞模擬段落。
    """
    df_raw = _read_csv_upload(file)
    if augment_high:
        df_raw = augment_with_high_fatigue(df_raw)

    df_proc = run_pipeline(df_raw)
    level_counts = df_proc["level"].value_counts().to_dict()
    sample = df_proc.head(5).to_dict(orient="records")

    return {
        "rows_processed": int(df_proc.shape[0]),
        "level_counts": level_counts,
        "columns": df_proc.columns.tolist(),
        "preview": sample,
    }


@app.post("/train")
def train_models(
    file: UploadFile = File(...),
    augment_high: bool = True,
) -> Dict[str, Any]:
    """
    上傳 CSV，執行疲勞管線並訓練 baseline 模型。

    回傳分類報告與燈號統計。
    """
    df_raw = _read_csv_upload(file)
    if augment_high:
        df_raw = augment_with_high_fatigue(df_raw)

    df_proc = run_pipeline(df_raw)
    result = build_and_train_baselines(df_proc, return_reports=True)

    if isinstance(result, tuple):
        models, reports = result
    else:
        models, reports = result, {}

    trained = sorted(models.keys())
    level_counts = df_proc["level"].value_counts().to_dict()

    return {
        "trained_models": trained,
        "reports": reports,
        "level_counts": level_counts,
        "rows_processed": int(df_proc.shape[0]),
    }
