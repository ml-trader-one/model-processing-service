import asyncio
import time
from datetime import datetime

import structlog
import mlflow
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.repository import fetch_candles
from app.ml_service import predict_next_close
from app.metrics import (
    SCHEDULER_JOB_RUNS_TOTAL,
    SCHEDULER_JOB_DURATION_SECONDS,
    SCHEDULER_JOBS_ACTIVE,
)

logger = structlog.get_logger(__name__)

# Инициализируем планировщик.
# В идеале подключить к БД, чтобы расписание выживало при перезапуске контейнера!
scheduler = AsyncIOScheduler()

# Кэш для загруженных моделей, чтобы не качать их из MLflow каждый раз
_MODEL_CACHE = {}


async def execute_scheduled_inference(instrument_uid: str, interval: str, run_id: str):
    """Задача, которая запускается по CRON расписанию"""
    logger.info("Starting scheduled inference", instrument=instrument_uid, interval=interval)

    # Обновляем количество активных джобов
    SCHEDULER_JOBS_ACTIVE.set(len(scheduler.get_jobs()))

    job_start = time.monotonic()
    status = "success"

    try:
        # 1. Достаем последние 100 свечей из базы
        df = await fetch_candles(
            instrument_uid=instrument_uid,
            interval=interval,
            limit=100,
        )

        # 2. Вызываем ваш метод в отдельном пуле потоков.
        # Декоратор @mlflow.trace отработает автоматически!
        result = await asyncio.to_thread(
            predict_next_close,
            interval=interval,
            df=df,
            instrument_uid=instrument_uid,
        )

        # 3. Логируем результат инференса как отдельный Run в MLflow
        # Это позволит вам строить красивые дашборды с результатами работы модели
        mlflow.set_experiment(f"Inference_{instrument_uid}_{interval}")

        run_name = f"pred_{datetime.now().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_used": result["model_used"],
                "prediction_date": result["prediction_date"],
                "last_candle_time": result["last_candle_time"],
                "train_run_id": run_id,  # Указываем, какая именно модель делала прогноз
            })

            mlflow.log_metrics({
                "predicted_close": result["predicted_close"],
                "last_close": result["last_close"],
                "change_pct": result["change_pct"],
            })

            if "confidence_interval" in result:
                mlflow.log_metrics({
                    "lower_90": result["confidence_interval"]["lower_90"],
                    "upper_90": result["confidence_interval"]["upper_90"],
                })

        logger.info("Scheduled prediction success", result=result)

        # TODO: Здесь можно сохранить результат в БД или отправить в Kafka

    except Exception as e:
        status = "error"
        logger.exception("Scheduled inference failed", error=str(e))

    finally:
        duration = time.monotonic() - job_start
        SCHEDULER_JOB_RUNS_TOTAL.labels(
            instrument_uid=instrument_uid, interval=interval, status=status
        ).inc()
        SCHEDULER_JOB_DURATION_SECONDS.labels(
            instrument_uid=instrument_uid, interval=interval
        ).observe(duration)
        SCHEDULER_JOBS_ACTIVE.set(len(scheduler.get_jobs()))