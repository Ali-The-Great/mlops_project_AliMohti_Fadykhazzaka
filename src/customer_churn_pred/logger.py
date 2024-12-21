from loguru import logger

def get_logger():
    logger.add("logs/{time}.log", rotation="1 week", level="INFO")
    return logger