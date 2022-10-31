# -*- encoding: utf-8 -*-
'''
@File    :   general.py    
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author 
------------      ------- 
2021/8/4 8:58   yjy      

@Desciption
'''

import json
from pathlib import Path


def checkWithMakeDir(path):
    """
    判断路径是否存在，不存在则创建, 并且
    :param path:
    :return: 直接返回一个 pathlib.Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# 保存文件
def saveFile(parent_path, name, content):
    parent_path = checkWithMakeDir(parent_path)
    with open(parent_path / name, mode='w', encoding='utf-8') as f:
        f.write(content)


# open cv 保存图片
def cvSaveImg(cv2, parent_path, name, img):
    parent_path = Path(checkWithMakeDir(parent_path))
    # open cv 只认 str 对象
    cv2.imwrite(str(parent_path / name), img)


# 保存json文件
def saveJsonFile(parent_path, name, content):
    parent_path = Path(checkWithMakeDir(parent_path))
    with open(parent_path / name, mode='w', encoding='utf-8') as f:
        json.dump(content, f)


def checkNone(dict, key):
    if not key in dict or dict[key] == None or dict[key] == '':
        raise Exception('参数错误，没有{0}'.format(key))
    return dict[key]
