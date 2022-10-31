# -*- encoding: utf-8 -*-
"""
@File    : misc.py
@Time    : 2020/12/14 13:13
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.


"""
import os
import functools

from common import consts
from utils import redis_tools
from werkzeug.exceptions import BadRequest
import uuid
from utils.log_util import run_log
from exceptions import *
import requests
import json
import numpy as np
from threading import Thread
from common.consts import java_conf
from fitow_celery.fitow_celery import logger

def filter_by_threshold(inference_result, threshold=.5):
    """ Postprocess inference result """
    tuples = zip(inference_result['detection_boxes'],
                 inference_result['detection_scores'],
                 inference_result['detection_classes'])
    return list(filter(lambda x: x[1] > threshold, tuples))


def _log_and_get_handler_error(error_message, status, response):
    """ Log and get handler error """
    # unexpected_error_logger.critical(error_message)
    run_log.exception(error_message)
    response["msg"] = 'an error occurred'
    response["code"] = status
    return response, status


def log_and_wait(proc, proc_name=None, logger=run_log):
    """ Log and wait a child process """

    class _ProcessUnexpectedlyQuitsException(Exception):
        pass

    if not proc_name:
        proc_name = str(proc)
    logger.debug('Wait child process %s(pid: %s) to terminate' % (proc_name, proc.pid))
    proc.wait()
    if proc.returncode != 0:
        raise _ProcessUnexpectedlyQuitsException(
            'Child process %s(pid: %d) unexpectedly quits. (returncode=%d)'
            % (proc_name, proc.pid, proc.returncode))
    logger.debug('Child process %s(pid: %s) terminated' % (proc_name, proc.pid))


def try_call(handle):
    """

    :param handle:
    :return:
    """

    functools.wraps(handle)

    def wrapper(*args, **kwargs):
        """ Wrapper """
        response = {"msg": "", "code": 200, "data": {}}

        try:
            data = handle(*args, **kwargs)
            response["data"] = data
        except (BadRequest,
                TrainingJobNotExistsException,
                InconsistentStateException,
                EmptyDatasetException) as e:

            response = _log_and_get_handler_error(e, 400, response)
        except Exception as e:
            response = _log_and_get_handler_error(e, 500, response)
        finally:
            run_log.info(f"Outer handler quits. (response={str(response)})")
            return response

    return wrapper


def get_uuid():
    """ UUID Generator """
    newuuid = uuid.uuid1()
    newuuid_str = str(newuuid)
    newuuid_str_no_hyphens = newuuid_str.replace("-", "")
    return newuuid_str_no_hyphens


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def callback_training_loss(global_step: int,
                           version_id: str = "",
                           total_loss: int = 0,
                           data_type: str = "Train",
                           precision = 0,
                           logger=logger):
    """

    return training losses and global step to java

    :param data_type:
    :param global_step: global step
    :param total_loss:  total loss
    :param version_id version id
    :param precision mAP of trainined model
    :return:
    """

    def call_back_func(global_step, version_id="", total_loss=0, data_type="Train", precision=0):
        data = {
            "dataType": "Train/Valid",
            "modelFileName": "",
            "modelLoss": 0,
            "modelPrecision": precision,
            "modelStep": 0,
            "trainTaskId": ""
        }

        data.update({
            "dataType": data_type,
            "modelLoss": total_loss,
            "modelPrecision": precision,
            "modelStep": global_step,
            "trainTaskId": version_id
        })

        # run_log.info(data)
        # java_url = "http://127.0.0.1:9800/api/qic/v1/model/task/train/loss/add"
        java_url = java_conf['host'] + "api/qic/v1/model/task/train/loss/add"

        headers = {'Content-Type': 'application/json'}

        logger.info("eval result is:{}".format(data))
        response = requests.post(url=java_url, headers=headers, data=json.dumps(data))
        return response

    # to avoid celery stuck
    post_process = Thread(target=call_back_func, args=(global_step, version_id, total_loss, data_type, precision))
    post_process.start()


def callback_ocr_metrics(step: int,
                         version_id: str = "",
                         total_loss: float = 0,
                         data_type: str = "Train",
                         precision: float = 0,
                         recall: float = 0,
                         accuracy: float = 0,
                         taskType: str = "",
                         logger=logger):
    """

    return training losses and global step to java

    :param data_type:
    :param global_step: global step
    :param total_loss:  total loss
    :param version_id version id
    :param precision mAP of trainined model
    :return:

    Args:
        accuracy:
        taskType:
        recall:
        accuracy:
        recall:
    """

    def call_back_func(global_step, version_id="", total_loss=0, data_type="Train", precision=0, recall=0, accuracy=0,taskType=""):
        data = {
            "dataType": "Train/Valid",
            "modelFileName": "",
            "modelLoss": 0,
            "modelPrecision": 0,
            "modelStep": 0,
            "taskId": "",
            "modelRecall": 0,
            "modelAccuracy": 0,
            "taskType": ""
        }

        data.update({
            "dataType": data_type,
            "modelLoss": total_loss,
            "modelPrecision": precision,
            "modelStep": global_step,
            "taskId": version_id,
            "modelRecall": recall,
            "modelAccuracy": accuracy,
            "taskType": taskType
        })
        # run_log.info(data)
        # java_url = "http://127.0.0.1:9800/api/qic/v1/model/task/train/loss/add"
        java_url = java_conf['host'] + "ocr/task/train/addLoss"

        headers = {'Content-Type': 'application/json'}

        logger.info("eval result is:{}".format(data))
        response = requests.post(url=java_url, headers=headers, data=json.dumps(data))
        return response

    # to avoid celery stuck
    post_process = Thread(target=call_back_func, args=(step, version_id, total_loss, data_type, precision, recall, accuracy,taskType))
    post_process.start()

def notify_complete(r_key: str = "",
                    url: str = '/ocr/dataset/preMarkResult',
                    code: int = 200,
                    msg: str = "",
                    logger=logger):

    # def call_back_func(r_key=""):
    data = {"key": r_key, 'code': code, 'msg': msg}


    java_url = java_conf['host'] + url

    headers = {'Content-Type': 'application/json'}

    logger.info(f"notify complete with key: {r_key}, {code}, {url}")
    response = requests.post(url=java_url, headers=headers, data=json.dumps(data))
    return response

    # to avoid celery stuck
    # post_process = Thread(target=call_back_func, args=(r_key))
    # post_process.start()

def eval_result_consumer(eval_result):
    """
    a consumer of to return the eval result to java

    :param eval_result: confusion matrix result dict like:

        final_res = {
                "detect_res":{
                    "name_list":["标签id","通过","误检","漏检","精度","召回","过杀"],
                    "result":chaos_result.tolist()
                              },
                "chaos_res":{
                    "name_list":self.columns,
                    "result":chaos_matrix.tolist()
                }
            }

    Caution : the mAP is mean average precision at threshold 0.2

    :return:
    """

    result_matrix = np.asarray(eval_result["detect_res"]["result"])
    presion = result_matrix[:, 4]
    recall = result_matrix[:, 5]

    AP = np.mean(presion)
    if AP is (np.inf or np.nan):
        AP = 0.0

    version_id = eval_result["version_id"]
    global_step = eval_result["current_step"]
    callback_training_loss(global_step, version_id, total_loss=0, data_type="Valid", precision=AP)


def filter_nan_or_inf(result_matrix_array):
    """
    filter nan or inf in matrix
    replace them to zeors

    :param result_matrix_list:
    :return:
    """
    result_matrix_array[np.isnan(result_matrix_array)] = 0
    result_matrix_array[np.isinf(result_matrix_array)] = 0

    return result_matrix_array


def dispatch_yolov5_engine(trainTaskId, modelFileId, modelVersion, device_ids, userId, userName, modelId):
    def call_back_func(trainTaskId, modelFileId, modelVersion, device_ids, userId, userName, modelId):
        data = {
            "trainTaskId": trainTaskId,
            "modelFileId": modelFileId,
            "modelVersion": modelVersion,
            "device_ids": device_ids,
            "userId": userId,
            "userName": userName,
            "modelId": modelId
        }

        java_url = java_conf['host'] + "api/qic/v1/model/task/train/data/dispatch"

        headers = {'Content-Type': 'application/json'}

        requests.post(url=java_url, headers=headers, data=json.dumps(data))

    try:
        post_process = Thread(target=call_back_func,
                              args=(trainTaskId, modelFileId, modelVersion, device_ids, userId, userName, modelId))
        post_process.start()
    except Exception as e:
        redis_tools.h_del(consts.DISPATCH_STATUS, trainTaskId)
        redis_tools.set_remove(True, consts.DISPATCHING_IDS, *device_ids)
        redis_tools.set_remove(True, consts.DISPATCHING_MODEL_IDS, modelId)
        raise e
