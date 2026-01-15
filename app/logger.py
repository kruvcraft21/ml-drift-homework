import logging

def get_logger(file_name: str, logger_name : str = "uvicorn.error") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s,%(message)s",
        "%Y-%m-%dT%H:%M:%S%z",
    ))
    logger.addHandler(file_handler)
    return logger