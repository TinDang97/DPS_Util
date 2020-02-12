import logging
import sys
import os

root_dir = 'logs/'
try:
    os.mkdir(root_dir)
except:
    pass


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
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    log.addHandler(ch)
    return log


info_log = get_logger(logger_name='info')
error_log = get_logger(logger_name='error')