# -*- encoding: utf-8 -*-

from common.consts import celery_config

# Local
# CELERY_RESULT_BACKEND = 'redis://localhost'
# BROKER_URL = 'redis://localhost'

# Online
broker_url = 'redis://:{password}@{host}:{port}/{dbnum}'.format(**celery_config['broker_url'])

result_backend = 'redis://:{password}@{host}:{port}/{dbnum}'.format(**celery_config['result_backend'])

task_track_started = True

worker_max_tasks_per_child = 1
task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Asia/Shanghai'
include = ['task_and_starter.tasks.start_train_task',
           'task_and_starter.tasks.start_test_task'
           ]

