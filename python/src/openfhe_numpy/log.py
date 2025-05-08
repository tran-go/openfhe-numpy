import inspect
import os
import logging
import sys

FP_ENABLE_DEBUG = os.getenv("FP_ENABLE_DEBUG", "OFF") == "ON"
logger = logging.getLogger("openfhe_numpy")

if not logger.hasHandlers():
    handler = logging.FileHandler("openfhe_numpy.log")
    # formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _format_log(level: str, message: str) -> str:
    frame = inspect.stack()[2]
    filename = os.path.basename(frame.filename)
    function_name = frame.function
    line_number = frame.lineno
    return f'{level}: {message}\n    File: "{filename}", line {line_number}, in {function_name}'


def FP_ERROR(message: str):
    message = _format_log("FP_ERROR", message)
    logger.debug(message)
    print(message)
    sys.exit(1)


def FP_DEBUG(message: str):
    if FP_ENABLE_DEBUG:
        message = _format_log("FP_DEBUG", message)
        logger.debug(message)
        print(message)


def FP_WARNING(message: str):
    message = _format_log("FP_WARNING", message)
    logger.debug(message)
    print(message)
