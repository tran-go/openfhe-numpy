import inspect
import os


def get_log_info():
    frame = inspect.stack()[2]  # two levels up
    filename = os.path.basename(frame.filename)
    function_name = frame.function
    return filename, function_name


def FP_DEBUG(msg):
    filename, function_name = get_log_info()
    print(f"[FP_DEBUG] [{filename}:{function_name}] {msg}")


def FP_ERROR(msg):
    filename, function_name = get_log_info()
    print(f"[FP_ERROR] [{filename}:{function_name}] {msg}")
