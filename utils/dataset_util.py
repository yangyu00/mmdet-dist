# -*- encoding: utf-8 -*-
"""
@File    : elos_res_to_via.py
@Time    : 2020/1/2 14:18
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2020 The Fitow Authors. All Rights Reserved.

    转换elos的输出结果为via，便于遗漏的缺陷重新标注

"""
import json
import os
import cv2
import numpy as np
from collections import OrderedDict
from utils.log_util import run_log


INFO_JSON ={
    "year": 2019,
    "version": "1",
    "description": "Exported using Annotator",
    "contributor": "",
    "url": "",
    "date_created": ""
  }
LICENSE = [{
      "id": 1,
      "name": "Unknown",
      "url": ""
    }]

def gen_category_list_from_file(lable_file_path):
    """
    只包含coco json 的最后的label信息的文件
    :param lable_file_path:
    :return:
    """

    with open(lable_file_path,"r") as f:
        label_content = json.load(f)

    return label_content

def gen_label_map_pbtxt(coco_label_file,label_map_save_path):
    with open(coco_label_file,"r") as f:
        json_categorise = json.load(f)["categories"]
    with open(label_map_save_path, 'w') as ft:
        for i in json_categorise:
            item = ['   id: {}'.format(i['id']), '   name: "{}"'.format(i['name'])]
            item_w = 'item {' + '\n' + str(item[0]) + '\n' + str(item[1]) + '\n' + '}' + '\n'
            ft.write(item_w)
    return json_categorise

def gen_category_list(class_list,super_cate="fitow"):
    final_list = []
    for index, name in enumerate(class_list):
        item_dict = gen_img_json_item_dic(
            id=index+1,
            name=name,
            supercategory=super_cate,
        )
        final_list.append(item_dict)
    return final_list

def gen_img_json_item_dic(**kargs):
    dic = OrderedDict()
    dic.update(**kargs)
    return dic

def gen_coco_labelfile(label_file_dir,exported_json_name,label_json):

    """
    label_json: a category label list file must be provide manually

    """

    file_list = os.listdir(label_file_dir)
    img_index =0
    ann_index = 0

    encoded_img_list = []
    encoded_annotation_list = []
    # categories_list = gen_category_list(CATEGORY)
    categories_list = gen_category_list_from_file(label_json)["categories"]

    for file in file_list:
        try:
            if file.endswith("txt"):
                print("processing {}".format(file))
                txt_file = os.path.join(label_file_dir,file)
                img_name = txt_file[:-7]+"jpg"
                print(img_name)
                img = cv2.imread(img_name)
                img_height,img_width,img_channel = img.shape
                # img_height =4100
                # img_width = 8192
                img_base_name = os.path.basename(img_name)

                encoded_img_list.append(
                    gen_img_json_item_dic(
                        id=int(img_index),
                        width=int(img_width),
                        height=int(img_height),
                        file_name=img_base_name,
                        license=1,
                        date_captured="",
                    )
                )

                with open(txt_file) as f:
                    result_list = json.load(f)
                for result in result_list:
                    if  result:
                        xmin = int(result["x"])
                        ymin = int(result["y"])
                        xmax = int(result["x"]+result["w"])
                        ymax = int(result["y"]+result["h"])
                        bbox_width = int(result["w"])
                        bbox_height = int(result["h"])
                        categories_idx = int(result["obj_id"])

                        encoded_annotation_list.append(
                            gen_img_json_item_dic(
                                id = int(ann_index),
                                image_id = str(img_index),
                                category_id = int(categories_idx),
                                segmentation = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax],
                                area = float(bbox_height*bbox_width),
                                bbox = [xmin,ymin,bbox_width,bbox_height],
                                iscrowd = 0,
                            )
                        )
                        ann_index +=1
                img_index+=1
        except Exception as e:
            print(f"{file} error")

    final_json_dict = gen_img_json_item_dic(
        info=INFO_JSON,
        images=encoded_img_list,
        annotations=encoded_annotation_list,
        licenses=LICENSE,
        # categories = CATEGORIES_LIST,
        categories=categories_list
    )

    json_str = json.dumps(final_json_dict)
    with open(exported_json_name, 'w') as json_file:
        json_file.write(json_str)

    return exported_json_name

if __name__=="__main__":

    data_dir = r"D:\dataset\fitow_iqi_data_demo\train/"

