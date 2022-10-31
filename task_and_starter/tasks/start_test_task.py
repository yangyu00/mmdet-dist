# -*- encoding: utf-8 -*-

import argparse
import copy
import os
import os.path as osp
from pathlib import Path
import time
import warnings

import mmcv
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

from fitow_celery.fitow_celery import app

from task_and_starter.tasks.base_task import AIBaseTask
from common.consts import BASE_WORK_DIR, BASE_DATA_DIR
from utils.eval_res import output_eval_res


@app.task(base=AIBaseTask)
def start_test(**opt):
    """
    version_id
    :return:
    """
    args = argparse.Namespace(**opt)

    args.work_dir = osp.join(BASE_WORK_DIR, args.work_dir_name)
    alist = list(Path(args.work_dir).rglob('*.py'))
    if alist:
        args.config = str(alist[0])
    else:
        raise ValueError('The work_dir must contain a config file.')
    args.checkpoint = osp.join(args.work_dir, args.selected_ckpt)

    if isinstance(args.img_scale, list):
        if isinstance(args.img_scale[0], int):
            args.input_img_scale = tuple(args.img_scale)

    args.cfg_options = {'data_root': BASE_DATA_DIR,
                        'data.test.ann_file': osp.join(BASE_DATA_DIR, 'annotations', args.test_ann_file),
                        'data.test.img_prefix': osp.join(BASE_DATA_DIR, args.test_ann_file.split('.')[0]),
                        'data.test.pipeline.1.img_scale': args.input_img_scale,
                        'data.test_dataloader.samples_per_gpu': args.samples_per_gpu,
                        'data.workers_per_gpu': args.workers_per_gpu
    }

    args.out, args.eval, args.format_only, args.show, args.show_dir = None, None, None, None, None

    for run_model in args.selected_running_model.split('_'):
        if 'eval' == run_model:
            args.eval = "bbox"
        if 'out' == run_model:
            args.out = osp.join(args.work_dir, 'result.pkl')

    # ==============================================================
    assert args.out or args.eval or args.format_only or args.show or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # ==============================================================
    config_path = args.config
    checkpoint_path = args.checkpoint

    ret = os.system(
        f'./mmdet_lib/dist_test.sh {config_path} {checkpoint_path} --work-dir {cfg.work_dir} --out {args.out} --eval {args.eval}'
    )

    assert int(ret >> 8) == 0, str(int(ret >> 8))

    # ==============================================================
    save_dir = osp.join(args.work_dir, 'eval_res')
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    output_eval_res(args.out, osp.join(BASE_DATA_DIR, 'annotations', args.test_ann_file), args.out_score_thr, save_dir)
