# -*- encoding: utf-8 -*-

from marshmallow import EXCLUDE
from webargs import fields
from webargs.flaskparser import use_args
from common.consts import task_keys
from utils import redis_tools
from utils.misc import get_uuid
from ..behavior import behavior_guide
from ..result import response_handle_run, response_handle_stop
from task_and_starter.mmdet_task_starter import MMdetTasksStarter

base_args = {
    "version_id": fields.Str(required=True),
    "selected_model": fields.Str(required=True),
    "is_two_stage_method": fields.Bool(required=True),
    "work_dir_name": fields.Str(required=True)
}

train_args = {
    'train_ann_file': fields.Str(required=True),
    "epoch": fields.Int(required=True),
    'samples_per_gpu': fields.Int(required=True),
    'learning_rate': fields.Float(required=True),
    "no_validate": fields.Bool(required=True),
    'val_ann_file': fields.Str(),
    'val_insert_interval': fields.Int(),
    'resume_from': fields.Str(),
    'workers_per_gpu': fields.Int(),
    'gpu_id': fields.Int(),
    "auto_scale_lr": fields.Bool(),
    'auto_resume': fields.Bool(),
    'launcher': fields.Str(),
    'seed': fields.Int(),
    'diff_seed': fields.Bool(),
    'deterministic': fields.Bool(),
    'img_scale': fields.List(fields.Int(required=True), required=True),
    # 'multi_img_scale': fields.List(fields.List(fields.Int())),
    'num_classes': fields.Int(required=True),
    'classes_names': fields.List(fields.Str(required=True), required=True),
    'eval_threshold': fields.Float(),
}
train_args.update(base_args)


@behavior_guide.route("/start_training", methods=["POST"])
@response_handle_run
@use_args(train_args, unknown=EXCLUDE)
def start_train(args):

    trainer = MMdetTasksStarter(args)

    training_task_id = get_uuid()
    trainer.train(training_task_id)

    task_res = {"training_task_id": training_task_id}

    return task_res


stop_args = {
    "version_id": fields.Str(required=True),
    "training_task_id": fields.Str(required=True)
}

@behavior_guide.route("/stop_training", methods=["POST"])
@response_handle_stop
@use_args(stop_args, unknown=EXCLUDE)
def stop_train(args):
    version_id = None
    try:
        stopper = MMdetTasksStarter(args)
        stopper.stop(args['training_task_id'])

        version_id = args['version_id']
        task_res = {"stoped_version_id": args['version_id'],
                    "stoped_training_task_id": args['training_task_id']
                    }
        return task_res
    finally:
        if not version_id:
            version_id = args['version_id']
        redis_tools.h_del(task_keys['task_train'], version_id)
