import asyncio
from datetime import datetime

import structlog
import mlflow
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.repository import fetch_candles
from app.ml_service import predict_next_close

logger = structlog.get_logger(__name__)

scheduler = AsyncIOScheduler()

_MODEL_CACHE = {}


async def execute_scheduled_inference(instrument_uid: str, interval: str, run_id: str):
    logger.info("Starting scheduled inference", instrument=instrument_uid, interval=interval)
    try:
        df = await fetch_candles(
            instrument_uid=instrument_uid,
            interval=interval,
            limit=100
        )

        result = await asyncio.to_thread(
            predict_next_close,
            interval=interval,
            df=df,
            instrument_uid=instrument_uid
        )

        logger.info("Scheduled prediction success", result=result)

    except Exception as e:
        logger.exception("Scheduled inference failed", error=str(e))