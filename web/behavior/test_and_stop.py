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
    "work_dir_name": fields.Str(required=True),
    "selected_running_model": fields.Str(required=True)
}

test_args = {
    'selected_ckpt': fields.Str(),
    'test_ann_file': fields.Str(),
    'samples_per_gpu': fields.Int(required=True),
    'workers_per_gpu': fields.Int(),
    'gpu_id': fields.Int(),
    'img_scale': fields.List(fields.Int(required=True), required=True),
    # 'multi_img_scale': fields.List(fields.List(fields.Int()))
    'out_score_thr': fields.Float()
}
test_args.update(base_args)


@behavior_guide.route("/start_testing", methods=["POST"])
@response_handle_run
@use_args(test_args, unknown=EXCLUDE)
def start_teat(args):
    tester = MMdetTasksStarter(args)

    testing_task_id = get_uuid()
    tester.test(testing_task_id)

    task_res = {"testing_task_id": testing_task_id}

    return task_res


stop_args = {
    "version_id": fields.Str(required=True),
    "testing_task_id": fields.Str(required=True)
}


@behavior_guide.route("/stop_testing", methods=["POST"])
@response_handle_stop
@use_args(stop_args, unknown=EXCLUDE)
def stop_test(args):
    version_id = None
    try:
        stopper = MMdetTasksStarter(args)
        stopper.stop(args['testing_task_id'])

        task_res = {"stoped_version_id": args('version_id'),
                    "stoped_testing_task_id": args['testing_task_id']
                    }
        return task_res
    finally:
        if not version_id:
            version_id = args('version_id')
        redis_tools.h_del(task_keys['task_test'], version_id)
