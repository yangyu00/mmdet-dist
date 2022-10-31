# -*- encoding: utf-8 -*-

from pathlib import Path

from task_and_starter.abs_main_tasks_starter import MainTasksStarter
from common.consts import task_keys

from utils import redis_tools


from task_and_starter.tasks.start_train_task import start_train
from task_and_starter.tasks.start_test_task import start_test


class MMdetTasksStarter(MainTasksStarter):

    def __init__(self, json_dict):
        super().__init__(json_dict)

    def train(self, training_task_id):
        redis_tools.h_set(task_keys['task_train'], self.json_dict['version_id'], training_task_id, True)

        start_train.apply_async(
            kwargs={
                'selected_model': self.json_dict.get('selected_model', 'faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'),
                'is_two_stage_method': self.json_dict.get('is_two_stage_method', True),
                'work_dir_name': self.json_dict.get('work_dir_name', 'work_dir'),
                'train_ann_file': self.json_dict.get('train_ann_file', 'train2017.json'),
                'val_ann_file': self.json_dict.get('val_ann_file', 'train2017.json'),
                'epoch': self.json_dict.get('epoch', 3),
                'samples_per_gpu': self.json_dict.get('samples_per_gpu', 1),
                'workers_per_gpu': self.json_dict.get('workers_per_gpu', 2),
                'learning_rate': self.json_dict.get('learning_rate', 0.0025),
                'no_validate': self.json_dict.get('no_validate', False),
                'val_insert_interval': self.json_dict.get('val_insert_interval', 999),
                'resume_from': self.json_dict.get('resume_from', None),
                'gpu_id': self.json_dict.get('gpu_id', 0),
                'auto_scale_lr': self.json_dict.get('auto_scale_lr', False),
                'auto_resume': self.json_dict.get('auto_resume', False),
                'launcher': self.json_dict.get('launcher', 'none'),
                'seed': self.json_dict.get('seed', None),
                'diff_seed': self.json_dict.get('diff_seed', False),
                'deterministic': self.json_dict.get('deterministic', False),
                'img_scale': self.json_dict.get('img_scale', [600, 600]),
                'num_classes': self.json_dict.get('num_classes', 80),
                'classes_names': self.json_dict.get('classes_names', ['angle', 'devil']),
                'eval_threshold': self.json_dict.get('eval_threshold', 0.6)
            },
            task_id=training_task_id
        )

    def test(self, testing_task_id):

        redis_tools.h_set(task_keys['task_test'], self.json_dict['version_id'], testing_task_id, True)

        start_test.apply_async(
            kwargs={
                'work_dir_name': self.json_dict.get('work_dir_name', 'work_dir'),
                'selected_ckpt': self.json_dict.get('selected_ckpt', 'latest.pth'),
                "selected_running_model": self.json_dict.get('selected_running_model', 'eval_out'),
                'test_ann_file': self.json_dict.get('test_ann_file', 'train2017.json'),
                'samples_per_gpu': self.json_dict.get('samples_per_gpu', 1),
                'workers_per_gpu': self.json_dict.get('workers_per_gpu', 0),
                'gpu_id': self.json_dict.get('gpu_id', 0),
                'img_scale': self.json_dict.get('img_scale', [600, 600]),
                'eval_options': self.json_dict.get('eval_options', None),
                'fuse_conv_bn': self.json_dict.get('fuse_conv_bn', False),
                'out_score_thr': self.json_dict.get('out_score_thr', 0.3),
                'show_score_thr': self.json_dict.get('show_score_thr', 0.3),
                'launcher': self.json_dict.get('launcher', 'none')
            },
            task_id=testing_task_id
        )
