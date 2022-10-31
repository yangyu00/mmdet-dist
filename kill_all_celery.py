# -*- encoding: utf-8 -*-

import os

import redis

from common.consts import java_conf

if __name__ == '__main__':
    r = redis.Redis(**java_conf['udev']['redis'], db=0)
    r.delete('unacked', 'unacked_index')
    r.close()

    os.system(
        "ps auxww | grep 'celery worker' | awk '{print $2}' | xargs kill -9 && pkill -9 -f 'fitow_celery'"
    )