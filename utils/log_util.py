# -*- encoding: utf-8 -*-
"""
@File    : log_util.py
@Time    : 2020/12/10 13:35
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.


"""
import logging
import logging.handlers
from common.consts import LOG_DIR
from pathlib import Path

BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
# e_n = 'error.log'
# r_n = 'run.log'
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


def set_log(log_name, e_n = 'error.log', r_n = 'run.log'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # error log
    e_h = logging.handlers.RotatingFileHandler(Path(LOG_DIR) / e_n,
                                               maxBytes=10 * 1024 * 1024,
                                               backupCount=5,
                                               encoding='utf-8')
    e_h.setLevel(logging.ERROR)
    e_h.setFormatter(formatter)

    # normal log
    n_h = logging.handlers.RotatingFileHandler(Path(LOG_DIR) / r_n,
                                               maxBytes=20 * 1024 * 1024,
                                               backupCount=5,
                                               encoding='utf-8')
    n_h.setLevel(logging.INFO)
    n_h.setFormatter(formatter)
    n_h.addFilter(LevelFilter(logging.ERROR))

    logger.addHandler(e_h)
    logger.addHandler(n_h)

    logger.propagate = False

    return logger


run_log = set_log("running_log")
