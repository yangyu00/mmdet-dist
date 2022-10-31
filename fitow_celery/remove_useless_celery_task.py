# -*- encoding: utf-8 -*-
"""
@File    : remove_useless_celery_task.py
@Time    : 2021/2/7 12:21
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2021 The Fitow Authors. All Rights Reserved.

    or  celery -A fitow_celery.fitow_celery purge

"""

from fitow_celery import app


def run():
    app.control.purge()

if __name__ =="__main__":
    run()