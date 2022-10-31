#!/usr/bin/env bash

CONFIG=$1
python -m torch.distributed.launch \
          $(dirname "$0")/train.py \
          $CONFIG \
          ${@:2}
if [ $? -eq 0 ]; then
    exit 0
else
    exit 1
fi