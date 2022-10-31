# -*- encoding: utf-8 -*-
"""
@File    : exceptions.py
@Time    : 2020/12/12 14:44
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.


"""
import traceback
from utils.log_util import set_log
import time

unexpect_log_date = time.strftime("%Y-%m-%d", time.localtime())
unexpected_error_logger = set_log(f"UEXPECTED_ERROR_{unexpect_log_date}")

class TrainingJobNotExistsException(Exception):
    pass

class StartTrainingException(Exception):
    pass

class InconsistentStateException(Exception):
    pass

class EmptyDatasetException(Exception):
    pass

class RequestDirNotFoundException(Exception):
    pass

class GenCocoFileException(Exception):
    pass

class GpuUnavailableException(Exception):
    pass

class LabelListNotExists(Exception):
    pass

class GenerateConfigException(Exception):
    pass

class UnexpectedException(Exception):
    pass

class PostParamException(Exception):
    pass

class RegistNotExists(Exception):
    pass

class YoloTrainingException(Exception):
    pass



def unexpected_error(e, hint='(No hint)', logger=unexpected_error_logger):
    """ Invoked when unexpected error occurred"""
    logger.error("UEXPECTED ERROR {} of {}".format(hint,str(e)))
    logger.error(traceback.print_exc())