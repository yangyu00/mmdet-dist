# -*- encoding: utf-8 -*-
"""
@File    :   base_task.py
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author
------------      -------
2022/1/22 14:26   yjy

@Description:

主要是会用到显卡的任务, 因为要释放显卡

"""
import sys

from exceptions import unexpected_error
from celery import Task

from utils import redis_tools
from common.consts import E_SFX

from fitow_celery.fitow_celery import logger


class AIBaseTask(Task):

    __EXP = 60*60*2

    def __revoke(self, task_id, task_name):
        logger.info('Kill task %s(task_id: %s)' % (task_name, task_id))
        try:
            signal = 'TERM'
            if sys.platform == "linux":
                signal = "SIGKILL"
            self.app.control.revoke(task_id, terminate=True, signal=signal)
        except Exception as e:
            unexpected_error(
                e, hint='celery failed to stop task %s(task_id: %s).' % (task_name, task_id))

    def on_success(self, retval, task_id, args, kwargs):
        """Success handler.

        Run by the worker if the task executes successfully.

        Arguments:
            retval (Any): The return value of the task.
            task_id (str): Unique id of the executed task.
            args (Tuple): Original arguments for the executed task.
            kwargs (Dict): Original keyword arguments for the executed task.

        Returns:
            None: The return value of this handler is ignored.
        """
        logger.info("++++++++++++++++++++++++++++")
        logger.info("this is on success")
        redis_tools.s_set(f'suc_{task_id}', 1, ex=self.__EXP)

        self.__revoke(task_id, self.name)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Error handler.

        This is run by the worker when the task fails.

        这里 revoke 状态就变为了 REVOKED，如果不在这里revoke，
        状态是 FAILURE，无法进行 revoke（不知道为什么，待研究）
        所以，在这里 revoke 前，redis里存一个值，表示当前的 revoked状态，
        实际是 failure，即报错产生的

        Arguments:
            exc (Exception): The exception raised by the task.
            task_id (str): Unique id of the failed task.
            args (Tuple): Original arguments for the task that failed.
            kwargs (Dict): Original keyword arguments for the task that failed.
            einfo (~billiard.einfo.ExceptionInfo): Exception information.

        Returns:
            None: The return value of this handler is ignored.
        """
        logger.info("++++++++++++++++++++++++++++")
        logger.info("this is on failure")
        logger.info(einfo)

        try:
            if exc.__str__().startswith('Unable to find a valid cuDNN algorithm to run convolution') \
                    or exc.__str__().startswith('CUDA out of memory'):

                redis_tools.s_set(task_id+E_SFX, 20001)

        finally:
            # 存入redis
            redis_tools.s_set(f'rev_fail_{task_id}', 1, ex=self.__EXP)
            self.__revoke(task_id, self.name)