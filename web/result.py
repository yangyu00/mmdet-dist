# -*- encoding: utf-8 -*-

from functools import wraps
from typing import Dict, Any

from marshmallow import ValidationError

from common.web_msg_code import m_c
from utils.log_util import run_log

return_type = Dict[str, Any]


def base_return(res: Dict[str, Any], code: int) -> return_type:
    return {'msg': m_c[code], 'code': code, 'data': res}


def success(res: Dict[str, Any]) -> return_type:
    return base_return(res, 200)


def err(e: Exception, code: int = 500) -> return_type:
    run_log.exception(e)
    run_log.error('====================================')
    return base_return({}, code)


def response_handle_run(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            res = f(*args, **kwargs)
            return success(res)
        except ValidationError as e:
            return err(e, 600)
        except RuntimeError as e:
            if e.__str__().startswith('CUDA out of memory') \
                    or e.__str__().startswith('Unable to find a valid cuDNN algorithm to run convolution'):
                return err(e, 20001)
            else:
                return err(e, 600)

        except Exception as e:
            return err(e)

    return wrapper


def response_handle_stop(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            res = f(*args, **kwargs)
            return success(res)
        except ValidationError as e:
            return err(e, 600)
        except Exception as e:
            return err(e)

    return wrapper

