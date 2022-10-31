# -*- encoding: utf-8 -*-
"""
@File    : stop_celery_training.py
@Time    : 2020/12/10 17:37
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.


"""

import os
from utils.log_util import run_log
from fitow_celery import fitow_celery
from exceptions.exceptions import unexpected_error
from utils import redis_tools
from celery.result import AsyncResult
from fitow_celery.fitow_celery import app
import sys



def revoke(task_id, task_name):
    run_log.info('Kill task %s(task_id: %s)' % (task_name, task_id))
    try:
        result = AsyncResult(id=task_id, app=app)
        if result.state == 'STARTED':
            running_info = result.result
            if isinstance(running_info, dict):
                running_pid = running_info.get("pid", [])

            if running_pid:
                os.system(f'./fitow_celery/kill_training.sh {running_pid}')

        signal='TERM'
        if sys.platform =="linux":
            signal = "SIGKILL"
        fitow_celery.app.control.revoke(task_id, terminate=True, signal=signal)


    except Exception as e:
        unexpected_error(
            e, hint='celery failed to stop task %s(task_id: %s).' % (task_name, task_id))



