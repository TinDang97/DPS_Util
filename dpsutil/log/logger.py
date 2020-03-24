import logging
import sys


def get_logger(logger_name='default'):
    """
    Get logging and format
    All logs will be saved into logs/log-DATE (default)
    Default size of log file = 15m
    :param logger_name:
    :return:
    """
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log_format = logging.Formatter("%(asctime)s - %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    log.addHandler(ch)
    return log


logger = get_logger(logger_name='info')
write_info = logger.info
write_warning = logger.warning
write_error = logger.error

__all__ = ['logger', 'get_logger', 'write_error', 'write_info', 'write_warning']
