#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
python -m torch.distributed.launch \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    ${@:3}
if [ $? -eq 0 ]; then
    exit 0
else
    exit 1
fi
