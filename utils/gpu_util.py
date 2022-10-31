# -*- encoding: utf-8 -*-
"""
@File    : gpu_util.py
@Time    : 2020/12/29 12:07
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.


"""
import subprocess
import re
from utils.log_util import run_log
import pynvml


def is_gpu_available(gpus):
    """ is gpu available ? """
    try:
        # gpus = [int(i.strip()) for i in gpus.split(",")]
        p = subprocess.Popen("nvidia-smi", shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        out = ""
        while p.poll() is None:
            out += p.stdout.readline().decode().strip() + "@newline@"
        usages = [int(i) for i in re.findall(r"\s+(\d+)%\s+Default", out)]
        memories = [int(i[0]) / int(i[1]) for i in re.findall(r"(\d+)MiB\s+/\s+(\d+)MiB", out)]
        run_log.debug('gpu usages: %s, memories: %s' % (usages, memories))

        assert len(usages) == len(memories)
        if max(gpus) < 0 or max(gpus) >= len(usages):
            run_log.error('unknown gpu %d, total %d' % (max(gpus), len(usages)))
            return False

        usage_available = all([usages[i] == 0 for i in gpus])
        memory_available = all([memories[i] < .1 for i in gpus])

        return usage_available and memory_available
    except Exception as e:
        run_log.error('unexpected error occurred, return true by default. (error: %s)' % str(e))
        return True


def get_one_available_gpu():
    gpu_list = get_available_gpus()
    if len(gpu_list) == 0:
        return None
    return gpu_list[0]


def get_available_gpus():
    gpu_list = []
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if len(info_list) == 0:  # >0 表示有进程占用
            gpu_list.append(i)
    pynvml.nvmlShutdown()
    return gpu_list
