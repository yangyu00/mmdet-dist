# -*- encoding: utf-8 -*-

import os

if __name__ == '__main__':

    # for i in range(WORKERS_NUM):
    #     pidfile = os.path.join(LOG_DIR, f'celery{i+1}.pid')
    #     logfile = os.path.join(LOG_DIR, f'celery{i+1}.log')
    #     worker_name = f'worker{i+1}@fitowai'
    #     os.system(
    #         f'celery multi start {worker_name} -A fitow_celery.fitow_celery \
    #          --pidfile="{pidfile}" --logfile="{logfile}" -P solo -Q queue_{i+1}'
    #     )

    os.system(
        f'celery -A fitow_celery.fitow_celery worker  --pidfile /home/yangyu/data/logs/qic_ai/celery.pid \
        --concurrency=5 -D'
    )

