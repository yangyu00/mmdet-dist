# -*- encoding: utf-8 -*-

from __future__ import absolute_import

import logging
import logging.handlers
from pathlib import Path

import celery
import sys

from celery.signals import after_setup_logger

from common.consts import LOG_DIR
from utils.log_util import LevelFilter

sys.path.append("./mmdet_lib/")

app = celery.Celery('fitow_celery.fitow_celery')
app.config_from_object('fitow_celery.celery_config')

logger = logging.getLogger(__name__)

@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    # error log
    e_h = logging.handlers.RotatingFileHandler(Path(LOG_DIR) / 'celery_error.log',
                                               maxBytes=10 * 1024 * 1024,
                                               backupCount=5,
                                               encoding='utf-8')
    e_h.setLevel(logging.ERROR)
    e_h.setFormatter(formatter)

    # normal log
    n_h = logging.handlers.RotatingFileHandler(Path(LOG_DIR) / 'celery_run.log',
                                               maxBytes=20 * 1024 * 1024,
                                               backupCount=5,
                                               encoding='utf-8')
    n_h.setLevel(logging.INFO)
    n_h.setFormatter(formatter)
    n_h.addFilter(LevelFilter(logging.ERROR))

    logger.addHandler(e_h)
    logger.addHandler(n_h)


if __name__ == '__main__':
    app.start()
    setup_loggers(logger)
