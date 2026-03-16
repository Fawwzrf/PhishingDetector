# src/mltools/shared/logging.py

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    log_level  : str = "INFO",
    log_dir    : str = "logs",
    experiment : str = "default",
) -> None:
    """
    Setup loguru untuk seluruh mltools package.
    Panggil sekali di awal setiap script/notebook.
    """
    logger.remove()

    # Console
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level    = log_level,
        colorize = True,
    )

    # File
    log_path = Path(log_dir) / experiment / "{time:YYYY-MM-DD}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format   = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        level    = "DEBUG",
        rotation = "50 MB",
        retention= "30 days",
        enqueue  = True,
    )