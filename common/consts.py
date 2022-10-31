# -*- encoding: utf-8 -*-
"""
@File    : consts.py
@Time    : 2020/12/10 13:32
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.


"""
celery_config = {
    "broker_url": {
        "password": "123456",
        "host": "127.0.0.1",
        "port": 6379,
        "dbnum": 0
    },
    "result_backend": {
        "password": "123456",
        "host": "127.0.0.1",
        "port": 6379,
        "dbnum": 0
    }
}
# java使用的redis库，udev
java_conf = {

    "udev": {
        "redis": {
            "password": "123456",
            "host": "127.0.0.1",
            "port": 6379
        },
    },
    "host": "http://127.0.0.1:9900/"
}


task_keys = {
    'task_train': 'task_train',
    'task_test': 'task_test',
    'task_detect': 'task_detect',
    'revoked_for_fail': 'revoked_for_fail'  # 因为任务失败而revoke的任务id
}

ONE_DAY = 60 * 60 * 24
# LOG_DIR = r"./logs"
LOG_DIR = r"/home/yangyu/data/logs/qic_ai/"
BASE_WORK_DIR = r"/home/yangyu/data/fitowai/result/"
BASE_DATA_DIR = r"/home/yangyu/data/fitowai/coco/"

VISIBLE_GPUS_FOR_TRAINING = '0'
VISIBLE_GPUS_FOR_EVALUATING = '0'
VISIBLE_GPUS_FOR_INFERRING = '-1'
MAX_GPU_MEMORY_FRACTION_FOR_TRAINING = .85
MAX_GPU_MEMORY_FRACTION_FOR_EVALUATING = .85

# 异步任务 task 错误信息 redis key 后缀
E_SFX = ':ecode'
