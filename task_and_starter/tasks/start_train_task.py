# -*- encoding: utf-8 -*-

import argparse

import os
import os.path as osp
from pathlib import Path

import warnings

import mmcv

from mmcv import Config
from mmcv.utils import config

from mmdet.utils import replace_cfg_vals, update_data_root

from fitow_celery.fitow_celery import app

from task_and_starter.tasks.base_task import AIBaseTask
from common.consts import BASE_WORK_DIR, BASE_DATA_DIR


@app.task(base=AIBaseTask)
def start_train(**opt):
    """
    version_id
    :return:
    """
    args = argparse.Namespace(**opt)

    args.config = osp.join(Path.cwd(), 'mmdet_lib', 'configs', args.selected_model)
    args.work_dir = osp.join(BASE_WORK_DIR, args.work_dir_name)

    if args.no_validate:
        args.val_insert_interval = 999
        args.val_ann_file = args.train_ann_file

    if isinstance(args.img_scale, list):
        if isinstance(args.img_scale[0], int):
            args.input_img_scale = tuple(args.img_scale)

    num_classes_route = 'model.roi_head.bbox_head.num_classes' if args.is_two_stage_method \
        else 'model.bbox_head.num_classes'
    workflow_changed = [('train', args.val_insert_interval)] if args.no_validate \
        else [('train', args.val_insert_interval), ('val', 1)]
    args.cfg_options = {'data_root': BASE_DATA_DIR,
                        'data.train.ann_file': osp.join(BASE_DATA_DIR, 'annotations', args.train_ann_file),
                        'data.train.img_prefix': osp.join(BASE_DATA_DIR, args.train_ann_file.split('.')[0]),
                        'data.train.pipeline.2.img_scale': args.input_img_scale,
                        'data.val.ann_file': osp.join(BASE_DATA_DIR, 'annotations', args.val_ann_file),
                        'data.val.img_prefix': osp.join(BASE_DATA_DIR, args.val_ann_file.split('.')[0]),
                        'data.val.pipeline.1.img_scale': args.input_img_scale,
                        'optimizer.lr': args.learning_rate,
                        'data.samples_per_gpu': args.samples_per_gpu,
                        'data.workers_per_gpu': args.workers_per_gpu,
                        'runner.max_epochs': args.epoch,
                        'workflow': workflow_changed,
                        'evaluation.interval': args.val_insert_interval,
                        num_classes_route: args.num_classes
    }

    if isinstance(args.classes_names, list):
        args.input_classes_names = tuple(args.classes_names)

    # ==============================================================
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.data.train.classes = args.input_classes_names  # 手动添加
    cfg.data.val.classes = args.input_classes_names    # 手动添加
    cfg.data.test.classes = args.input_classes_names   # 手动添加
    cfg.data.train.pipeline.append(config.ConfigDict({'type': 'FitowArgs'}))
    cfg.data.train.pipeline[-1].eval_threshold = 0.83

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # ==============================================================
    config_path = osp.join(cfg.work_dir, osp.basename(args.config))

    if args.no_validate:
        ret = os.system(
            f'./mmdet_lib/dist_train.sh {config_path} --work-dir {cfg.work_dir} --no-validate'
        )
    else:
        ret = os.system(
            f'./mmdet_lib/dist_train.sh {config_path} --work-dir {cfg.work_dir}'
        )

    assert int(ret >> 8) == 0, str(int(ret >> 8))

