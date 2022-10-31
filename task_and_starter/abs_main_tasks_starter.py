# -*- encoding: utf-8 -*-

from abc import abstractmethod, ABCMeta

from core.celery_stopers import CeleryStoper
from utils.log_util import run_log


class MainTasksStarter(metaclass=ABCMeta):

    def __init__(self, json_dict):
        self.json_dict = json_dict

    # @abstractmethod
    def train(self, training_task_id, evaluating_task_id):
        raise NotImplementedError

    # # @abstractmethod
    # def train_and_eval(self, training_task_id, evaluating_task_id):
    #     raise NotImplementedError

    def stop(self, training_task_id, evaluating_task_id=None):
        stopper = CeleryStoper(self.json_dict)
        stopper.stop(training_task_id, evaluating_task_id)
        run_log.info("stopped model")

    # # @abstractmethod
    # def test_on_images(self):
    #     raise NotImplementedError
    #
    # # @abstractmethod
    # def online_infer(self):
    #     raise NotImplementedError
