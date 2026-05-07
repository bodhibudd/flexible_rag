import sys

import loguru

def get_logger(level: str = "INFO", console: bool = True, logger_file: str = None):
    """

    :param level: 选择日志的级别，可选trace，debug，info，warning，error，critical
    :param console: 是不进行控制台输出日志
    :param logger_file: 日志文件路径，None则表示不输出日志到文件
    :return:
    """
    logger = loguru.logger
    logger.remove()

    if console:
        logger.add(sys.stderr, level=level.upper())

    # 添加一个文件输出的内容
    # 目前每天一个日志文件，日志文件最多保存7天
    if logger_file is not None:
        logger.add(
            logger_file,
            enqueue=True,
            level=level.upper(),
            encoding="utf-8",
            rotation="00:00",
            retention="7 days",
        )

    return logger

logger = get_logger(level="INFO", logger_file="server.log")