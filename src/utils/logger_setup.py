import logging
import os
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Redirects standard logging messages to Loguru.
    This is crucial for capturing logs from libraries like uvicorn or sqlalchemy.
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(debug: bool = False):
    """
    Configures Loguru. If debug is True, log_level is forced to DEBUG.
    """
    logger.remove()

    # Determine level: Runtime flag takes priority over Environment Variable
    if debug:
        log_level = "DEBUG"
    else:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    is_dev = os.getenv("APP_ENV", "development").lower() == "development"
    is_atty = sys.stderr.isatty()

    # Add the primary handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=is_atty,
        enqueue=True,
        backtrace=True,
        diagnose=is_dev or debug,  # Enable detailed error diagnosis in debug mode
    )

    # Suppress noise from frameworks
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.info(f"Logger initialized at level: {log_level}")
