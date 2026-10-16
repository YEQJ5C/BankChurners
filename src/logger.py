"""Structured JSON logger for production MLOps pipeline."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import sys

from src.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        dt = datetime.fromtimestamp(
            record.created,
            tz=datetime.now().astimezone().tzinfo,
        )
        timestamp_str = dt.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )

        log_data = {
            "ts": timestamp_str,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        skip_keys = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

        for key, value in record.__dict__.items():
            if key not in skip_keys:
                log_data[key] = value

        return json.dumps(
            log_data,
            ensure_ascii=False,
        )


def get_logger(
    name: str,
) -> logging.Logger:
    """Create structured JSON logger."""
    logger = logging.getLogger(name)

    lvl_name = settings.log_level.upper()
    level = getattr(
        logging,
        lvl_name,
        logging.INFO,
    )
    logger.setLevel(level)

    if not logger.handlers:
        settings.logs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        c_handler = logging.StreamHandler(
            sys.stdout,
        )
        c_handler.setFormatter(
            JSONFormatter(),
        )
        logger.addHandler(c_handler)

        f_path = (
            settings.logs_dir
            / "pipeline.jsonl"
        )
        f_handler = logging.FileHandler(
            f_path,
            encoding="utf-8",
        )
        f_handler.setFormatter(
            JSONFormatter(),
        )
        logger.addHandler(f_handler)

    logger.propagate = False
    return logger
