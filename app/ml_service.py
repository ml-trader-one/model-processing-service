from typing import Optional

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


def _model_key(instrument_uid: str, interval: str) -> str:
    return f"{instrument_uid}__{interval}"


def _model_path(instrument_uid: str, interval: str) -> Path:
    return Path(settings.models_dir) / _model_key(instrument_uid, interval)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["body"] = df["close"] - df["open"]
    df["shadow_upper"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["shadow_lower"] = df[["close", "open"]].min(axis=1) - df["low"]
    df["hl_spread"] = df["high"] - df["low"]
    df["body_pct"] = df["body"] / (df["open"] + 1e-9)

    for w in settings.ma_windows:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
        df[f"ma_{w}_dist"] = (df["close"] - df[f"ma_{w}"]) / (df[f"ma_{w}"] + 1e-9)

    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(settings.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(settings.rsi_period).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-9)

    for lag in [1, 2, 3, 5]:
        df[f"ret_{lag}d"] = df["close"].pct_change(lag)

    return df.dropna().reset_index(drop=True)


KNOWN_COVARIATES = [
    "open", "high", "low", "volume",
    "body", "shadow_upper", "shadow_lower", "hl_spread",
    "rsi", "macd", "bb_pct",
    "vol_ratio", "ret_1d", "ret_2d", "ret_5d",
]


def train_model(
        interval: str,
        df: pd.DataFrame,
        instrument_uid: str = "e6123145-9665-43e0-8413-cd61b8aa9b13",
        time_limit: Optional[int] = None,
) -> dict:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError:
        raise RuntimeError(
            "AutoGluon not installed. Run pip install autogluon.timeseries"
        )

    if len(df) < settings.min_candles_train:
        raise ValueError(
            f"Not enough candles to train: {len(df)} < {settings.min_candles_train}. "
        )

    df = add_technical_features(df)

    df["item_id"] = instrument_uid
    df = df.rename(columns={"time": "timestamp", settings.target: "target"})

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df[["item_id", "timestamp", "target"] + KNOWN_COVARIATES],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    path = _model_path(instrument_uid, interval)
    time_limit = time_limit or settings.training_time_limit

    predictor = TimeSeriesPredictor(
        path=str(path),
        prediction_length=settings.prediction_length,
        target="target",
        known_covariates_names=KNOWN_COVARIATES,
        eval_metric="MAPE",
        freq=interval,
    )
    predictor.fit(ts_df, time_limit=time_limit, presets="medium_quality")

    leaderboard = predictor.leaderboard(ts_df, silent=True)
    best_model = leaderboard.iloc[0]["model"]
    best_mape = float(leaderboard.iloc[0]["score_val"])

    meta = {
        "instrument_uid": instrument_uid,
        "interval": interval,
        "trained_at": datetime.now().isoformat(),
        "training_samples": len(df),
        "best_model": best_model,
        "mape": best_mape,
        "known_covariates": KNOWN_COVARIATES,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "candles_from": df["timestamp"].min().isoformat(),
        "candles_to": df["timestamp"].max().isoformat(),
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"[{instrument_uid} / {interval}] Train ended. "
        f"Best model: {best_model}, MAPE: {best_mape:.4f}"
    )
    return meta


def predict_next_close(
        interval: str,
        df: pd.DataFrame,
        instrument_uid: str = "e6123145-9665-43e0-8413-cd61b8aa9b13",
) -> dict:
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    except ImportError:
        raise RuntimeError("AutoGluon not installed.")

    path = _model_path(instrument_uid, interval)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: instrument_uid={instrument_uid!r}, interval={interval!r}. "
            "First the model should be trained with POST /train"
        )

    with open(path / "meta.json") as f:
        meta = json.load(f)

    if len(df) < settings.min_candles_predict:
        raise ValueError(
            f"Not enough candles for predictions: {len(df)} < {settings.MIN_CANDLES_PREDICT}."
        )

    df = add_technical_features(df)

    last_close = float(df["close"].iloc[-1])
    last_candle_time = df["time"].iloc[-1]

    df["item_id"] = instrument_uid
    df = df.rename(columns={"time": "timestamp", settings.target: "target"})

    ts_df = TimeSeriesDataFrame.from_data_frame(
        df[["item_id", "timestamp", "target"] + KNOWN_COVARIATES],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    predictor = TimeSeriesPredictor.load(str(path))
    predictions = predictor.predict(ts_df)
    predicted_close = float(predictions["mean"].iloc[0])

    next_dt = last_candle_time + timedelta(days=1)
    if interval == "1d":
        while next_dt.weekday() >= 5:
            next_dt += timedelta(days=1)

    change_pct = ((predicted_close - last_close) / last_close) * 100

    result = {
        "instrument_uid": instrument_uid,
        "interval": interval,
        "predicted_close": round(predicted_close, 4),
        "prediction_date": next_dt.strftime("%Y-%m-%d"),
        "last_close": round(last_close, 4),
        "last_candle_time": last_candle_time.isoformat(),
        "change_pct": round(change_pct, 2),
        "model_used": meta["best_model"],
    }

    if "0.1" in predictions.columns and "0.9" in predictions.columns:
        result["confidence_interval"] = {
            "lower_90": round(float(predictions["0.1"].iloc[0]), 4),
            "upper_90": round(float(predictions["0.9"].iloc[0]), 4),
        }

    return result
