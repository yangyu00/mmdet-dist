# -*- encoding: utf-8 -*-

from fitow_celery.stop_celery_training import revoke


class CeleryStoper:

    def __init__(self, requests_json_dict):
        self.version_id = requests_json_dict.get("version_id")

    def stop(self, training_task_id, evaluating_task_id=None):

        if training_task_id:
            revoke(training_task_id, 'training-task')

        if evaluating_task_id:
            revoke(evaluating_task_id, 'evaluating-task')

        return training_task_id, evaluating_task_id

