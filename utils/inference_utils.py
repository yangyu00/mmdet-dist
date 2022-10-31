# -*- encoding: utf-8 -*-
"""
@File    : create_fitow_tfrecord.py
@Time    : 2019/9/6 14:24
@Author  : XXX
@Email   : XXXX@fitow-alpha.com
@Software: PyCharm
# Copyright 2019 The Fitow Authors. All Rights Reserved.

Please note that this tool creates sharded output files.
"""
# 自动标注，导出DT文件
import json
from pathlib import Path

import numpy as np
import os
import time
import cv2
import sys
from copy import deepcopy

sys.getdefaultencoding()
sys.path.append("../model_lib/tf_od_v1")
sys.path.append("../model_lib/yolov5/v6")

from common.consts import LABEL_MAPPBTXT_FILE, ANNOTATION_LABEL_FILE_NAME, DT_NAME, GT_NAME, CATEGORY_LABEL_LIST_NAME
from utils.misc import filter_nan_or_inf


def eval_on_dataset(DT_dir, GT_dir, save_dir):
    """
    在数据集上推理并返回对应的推理结果，将结果写入文件

    :param DT_dir: DT路径，DT_dict.json
    :param GT_dir:  标签路径，via_export_coco.json
    :return: 返回一个json 字典,如下:
        {
         image_name:
         image_id:
         annotations:[
        {
         ground_truth:{
            {"id": 0,
            "image_id": "0",
            "category_id": 1,
            "category_name": "kp"
            "segmentation": [ 818, 448, 937, 448, 937, 696, 818, 696 ],
            "bbox": [ 818, 448, 119, 248 ],
            "iscrowd": 0}
         predictions:{
            {"id": 0,
            "image_id": "0",
            "category_id": 1,
            "category_name": "kp"
            "segmentation": [ 818, 448, 937, 448, 937, 696, 818, 696 ],
            "bbox": [ 818, 448, 119, 248 ],
            "iscrowd": 0}
         }
        }
    },]
}
    一个图像保存一个json结果，结果保存在savedir当中
    """

    Detection_dict = json.load(open(DT_dir, 'rb'))
    coco_json = json.load(open(GT_dir, 'rb'))
    categories = coco_json["categories"]
    img_w = coco_json["images"][0]["width"]
    img_h = coco_json["images"][0]["height"]
    category_id = {}
    for i, each_category in enumerate(categories):
        category_id[str(i + 1)] = each_category["name"]
    # print(category_id)

    # ————————————遍历标签，建立与图像索引匹配的列表，每个元素是一张图下的所有ground_truth————————————#
    annotation_images = [[] for i in range(coco_json["images"][-1]["id"] + 1)]
    annotations = coco_json["annotations"]  # 所有标签
    for i, each_anno in enumerate(annotations):  # 遍历coco_json的每个框，与image匹配
        each_anno["category_name"] = coco_json["categories"][each_anno["category_id"] - 1]["name"]
        annotation_images[int(each_anno["image_id"])].append(each_anno)
    for i in annotation_images:
        if i == []:
            annotation_images.remove(i)
    # ———————————————————————————————————————#
    predictions_images = []
    img_count = 1
    for key, each_image_dt in Detection_dict.items():
        DT_count = 0
        predictions_per_image = []
        for each_DT in each_image_dt:
            predictions = {"id": None, "image_id": None, "category_id": None, "category_name": None,
                           "segmentation": None, "bbox": None, "iscrowd": 0}
            each_category = str(each_DT[0])
            y_min = round(each_DT[1] * img_h)
            x_min = round(each_DT[2] * img_w)
            y_max = round(each_DT[3] * img_h)
            x_max = round(each_DT[4] * img_w)
            segmentation = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            predictions["image_id"] = str(img_count - 1)
            predictions["category_id"] = int(each_category)
            predictions["category_name"] = category_id[each_category]
            predictions["segmentation"] = segmentation
            predictions["bbox"] = bbox
            predictions["id"] = DT_count
            predictions_per_image.append(predictions)
            DT_count += 1
        predictions_images.append(predictions_per_image)
        img_count += 1
    image_names_to_pictureId = {}
    image_info = coco_json["images"]
    for i in image_info:
        image_names_to_pictureId[i["file_name"]] = i["pictureId"]

    DT_image_info = [key for key, value in Detection_dict.items()]
    for i, each_image in enumerate(DT_image_info):
        save_json_stracture = {"image_name": None, "image_id": None, "pictureId": None,
                               "annotations": [{"ground_truth": None, "predictions": None}]}
        save_json_stracture["image_name"] = each_image
        save_json_stracture["pictureId"] = image_names_to_pictureId[each_image]
        save_json_stracture["image_id"] = i
        save_json_stracture["annotations"][0]["ground_truth"] = annotation_images[i]
        save_json_stracture["annotations"][0]["predictions"] = predictions_images[i]
        # print(save_json_stracture)
        with open(os.path.join(save_dir, coco_json["images"][i]["file_name"][:-4] + ".json"), 'w') as f:
            json.dump(save_json_stracture, f, ensure_ascii=False)


class GT_Dict():
    """
    make groundtruth dict file
    """

    def __init__(self, via_file, columns, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.category = [{"id": i + 1, "name": col} for i, col in enumerate(columns)]
        self.via_file = via_file

    # 将json文件中的category项复制至此
    def category(self):

        category_id = {}
        category_id_r = {}
        list_ = []
        for each_category in self.category:
            id = each_category['id']
            name = each_category['name']
            category_id[id] = name
            list_.append(name)
        for each_category in self.category:
            id = each_category['id']
            name = each_category['name']
            category_id_r[name] = id
        return category_id, category_id_r, list_

    def img_names(self, via_coco_file):
        via_coco = json.load(open(via_coco_file, 'rb'))
        images_info_from_via = via_coco['images']
        img_name_dict = {}
        for i in images_info_from_via:
            img_name_dict[i["file_name"]] = i["id"]
        print(img_name_dict)
        return img_name_dict

    def anno_info(self, via_coco_file):
        via_coco = json.load(
            open(via_coco_file, 'rb'))
        anno_list = []
        anno_info_from_via = via_coco["annotations"]
        for i, each_anno in enumerate(anno_info_from_via):
            each_anno_dict = {}
            each_image_id = each_anno["image_id"]
            each_cate_id = each_anno["category_id"]
            each_box = each_anno["bbox"]
            x_min = each_box[0]
            y_min = each_box[1]
            w = each_box[2]
            h = each_box[3]
            each_box_norm = [(y_min) / self.img_h, (x_min) / self.img_w, (y_min + h) / self.img_h,
                             (x_min + w) / self.img_w]
            each_anno_dict['image_id'] = each_image_id
            each_anno_dict["category_id"] = each_box_norm
            each_anno_dict["category_id"].insert(0, each_cate_id)
            anno_list.append(each_anno_dict)
        return anno_list

    def make_anno_info_per_img(self):

        GT_dict = {}
        filenames = os.listdir(self.via_file)
        for filename in filenames:
            if filename.endswith(ANNOTATION_LABEL_FILE_NAME):
                via_coco_file = os.path.join(self.via_file, filename)
                print("读取：{}".format(filename))
                img_name_dict = self.img_names(via_coco_file)
                print("图片数：{}".format(len(img_name_dict)))
                for key, value in GT_dict.items():
                    if key in img_name_dict.keys():
                        print("发现重复图片：{}".format(key))
                anno_list = self.anno_info(via_coco_file)
                for img_name, img_id in img_name_dict.items():
                    anno_per_img = []
                    for each_anno in anno_list:
                        each_anno_list = {}
                        if each_anno['image_id'] == str(img_id):
                            anno_per_img.append(each_anno["category_id"])
                    GT_dict[img_name] = anno_per_img
        return GT_dict


class Auto_inference_to_DT():

    def __init__(self, infer_model, data_dir, output_dir, threshold=0.5):

        self.path_to_labels = os.path.join(data_dir, LABEL_MAPPBTXT_FILE)
        self.label_list_file = os.path.abspath(os.path.join(os.path.dirname(data_dir), CATEGORY_LABEL_LIST_NAME))

        self.infer_model = infer_model
        self.coco_json_path = os.path.join(data_dir, ANNOTATION_LABEL_FILE_NAME)
        self.path_to_test_images_dir = os.path.join(data_dir, "image")  # 预测的图片文件路径，如果是测试集就是测试集的所有图片
        self.confidence_score = threshold
        self.category_id, self.columns, self.category_num = self.gen_category_id()
        self.img_w, self.img_h = self.get_img_size()
        self.output_path = output_dir
        self.data_dir = data_dir

    def get_img_size(self):
        coco_json_path = self.coco_json_path
        if os.path.exists(coco_json_path):
            coco_json = json.load(open(coco_json_path, 'rb'))
            img_w = coco_json["images"][0]["width"]
            img_h = coco_json["images"][0]["height"]
            return img_w, img_h
        else:
            filename = os.listdir(self.path_to_test_images_dir)
            for file in filename:
                if file.endswith(".jpg" or ".bmp"):
                    img = cv2.imread(os.path.join(self.path_to_test_images_dir, file))
                    img_w, img_h = img.shape[1], img.shape[0]
                    return img_w, img_h

    def gen_category_id(self):

        """
        :return:
            category_id = {1:"kp"}
            columns = ["kp"]
            category_num
        """
        category_id = {}
        columns = []
        with open(self.label_list_file, "r") as f:
            json_categories = json.load(f)["categories"]

        for index, cat in enumerate(json_categories):
            cat_id = cat.get("id")
            cat_name = cat.get("name")
            category_id.update({cat_id: cat_name})
            columns.append(cat_name)

        category_num = len(json_categories)
        return category_id, columns, category_num

    def get_filename_list(self):
        pb_name_list = []
        pbtxt_name_list = []

        file_names = os.listdir(self.path_to_test_images_dir)
        for each_file in file_names:
            if each_file.endswith(".pb"):
                pb_name_list.append(each_file)
            elif each_file.endswith(".pbtxt"):
                pbtxt_name_list.append(each_file)
        return pb_name_list, pbtxt_name_list

    def inference_on_image(self, gpus="-1"):

        save_index = 0  # 保存图像用的记号

        DT_dict = {}

        total_images = len(os.listdir(self.path_to_test_images_dir))
        infer_all_start_time = time.time()
        for image_path in os.listdir(self.path_to_test_images_dir):
            each_DT_category_list = []
            if image_path.endswith(".jpg"):
                t_begin = time.time()
                # 这里根据文件名，获取GT中的字典信息：detections_boxes,category_ids
                image_name = os.path.join(self.path_to_test_images_dir, image_path)
                save_name = image_path  # 图片名
                image = cv2.imread(image_name)
                #################################输出DT_dict到json文件####################################

                DT_dict[save_name] = {}
                output_dict = self.infer_model.infer_single_image(image)

                detection_scores = output_dict["detection_scores"].tolist()
                detection_classes = output_dict["detection_classes"].tolist()
                detection_boxes = output_dict["detection_boxes"].tolist()

                detection_scores = [i for i in detection_scores if i >= float(self.confidence_score)]  # DT置信度
                detection_boxes = detection_boxes[:len(detection_scores)]  # DT 框坐标
                detection_classes = detection_classes[:len(detection_scores)]  # DT分类类别
                # {图片名字: {"类别名"：[bbox,score] , "类别名":[bbox,score] } ,{图片名字: {"类别名"：[bbox,score], "类别名": [classes,score]}
                for i, each_category in enumerate(detection_classes):
                    boxes_and_scores = detection_boxes[i]
                    boxes_and_scores.append(detection_scores[i])
                    boxes_and_scores.insert(0, each_category)
                    each_DT_category_list.append(boxes_and_scores)  # 图片名字: {"类别名"：[bbox,score] , "类别名":[bbox,score]
                    DT_dict[save_name] = (each_DT_category_list)

                save_index += 1
                print("图片序号:", save_index)
        infer_all_end_time = time.time()
        infer_all_time = infer_all_end_time - infer_all_start_time
        per_img_time = infer_all_time / total_images
        with open(os.path.join(self.output_path, DT_NAME), 'w') as f:
            json.dump(DT_dict, f)

        self.infer_model.close()
        print("预测图片输出完成，正在计算误检/漏检...")

        GT_Dict_maker = GT_Dict(self.data_dir, self.columns, self.img_w, self.img_h)
        GT_dict = GT_Dict_maker.make_anno_info_per_img()
        with open(os.path.join(self.output_path, "GT_dict.json"), 'w') as f:
            json.dump(GT_dict, f)

        final_res = {"number_images": total_images,
                     "all_infer_time": infer_all_time,
                     "per_image_infer_time": per_img_time,
                     "detection_info": DT_dict, "groundtruth_info": GT_dict}

        return final_res

    def inference_on_image_for_yolov5(self, dataset, gpus="-1"):

        save_index = 0  # 保存图像用的记号

        DT_dict = {}

        total_images = len(os.listdir(self.path_to_test_images_dir))
        infer_all_start_time = time.time()
        for image_path, img, im0s, _ in dataset:
            # for image_path in os.listdir(self.path_to_test_images_dir):
            each_DT_category_list = []
            if image_path.endswith(".jpg"):
                t_begin = time.time()
                # 这里根据文件名，获取GT中的字典信息：detections_boxes,category_ids
                image_name = os.path.join(self.path_to_test_images_dir, image_path)
                save_name = Path(image_path).name  # 图片名
                # image = cv2.imread(image_name)
                #################################输出DT_dict到json文件####################################

                DT_dict[save_name] = {}
                output_dict = self.infer_model.infer_single_image(img, im0s)

                detection_scores = output_dict["detection_scores"]
                detection_classes = output_dict["detection_classes"]
                detection_boxes = output_dict["detection_boxes"]

                detection_scores = [i for i in detection_scores if i >= float(self.confidence_score)]  # DT置信度
                detection_boxes = detection_boxes[:len(detection_scores)]  # DT 框坐标
                detection_classes = detection_classes[:len(detection_scores)]  # DT分类类别

                # if len(detection_classes) == 0:
                #     DT_dict[save_name] = ([])
                # else:
                # {图片名字: {"类别名"：[bbox,score] , "类别名":[bbox,score] } ,{图片名字: {"类别名"：[bbox,score], "类别名": [classes,score]}
                for i, each_category in enumerate(detection_classes):
                    boxes_and_scores = detection_boxes[i]
                    boxes_and_scores.append(detection_scores[i])
                    boxes_and_scores.insert(0, each_category)
                    each_DT_category_list.append(boxes_and_scores)  # 图片名字: {"类别名"：[bbox,score] , "类别名":[bbox,score]
                    DT_dict[save_name] = (each_DT_category_list)

                save_index += 1
                print("图片序号:", save_index)
        infer_all_end_time = time.time()
        infer_all_time = infer_all_end_time - infer_all_start_time
        per_img_time = infer_all_time / total_images
        with open(os.path.join(self.output_path, DT_NAME), 'w') as f:
            json.dump(DT_dict, f)

        self.infer_model.close()
        print("预测图片输出完成，正在计算误检/漏检...")

        GT_Dict_maker = GT_Dict(self.data_dir, self.columns, self.img_w, self.img_h)
        GT_dict = GT_Dict_maker.make_anno_info_per_img()
        with open(os.path.join(self.output_path, "GT_dict.json"), 'w') as f:
            json.dump(GT_dict, f)

        final_res = {"number_images": total_images,
                     "all_infer_time": infer_all_time,
                     "per_image_infer_time": per_img_time,
                     "detection_info": DT_dict, "groundtruth_info": GT_dict}

        return final_res

    def generate_chaos_matrix(self, current_step, version_id):

        GT_dict_path = os.path.join(self.output_path, GT_NAME)
        DT_dict_path = os.path.join(self.output_path, DT_NAME)
        if not os.path.exists(GT_dict_path):
            print("不存在GT信息，模式为预测模式，结束预测")
        else:
            print("找到GT数据，开始处理量子矩阵...")
            with open(DT_dict_path, 'rb') as dt_f:
                Detection_dict = json.load(dt_f)  # DT字典，需接在inference后
            with open(GT_dict_path, 'rb') as gt_f:
                Ground_Truth_dict = json.load(gt_f)  # GT字典

            chaos_result = np.zeros(
                (self.category_num, 7))  # 建立一个矩阵存放结果：分别是category_id[0]、通过[1]、误检[2]、漏检[3]、精度[4]、召回[5]
            img_path_dict = {}
            for i in self.columns:
                img_path_dict[i] = {
                    "Error": [],
                    "Miss": [],
                    'overfit': []
                }  # 建立一个字典用于存放每一类的误检和漏检图片路径，存放格式img_path_dict[category_id]["Error"]
            for i in range(len(chaos_result)):
                chaos_result[i][0] = i + 1
            chaos_matrix = np.zeros((self.category_num, self.category_num))  # 建立混淆矩阵
            OK_count = 0
            Error_count = 0
            Miss_count = 0
            Overfit_count = 0
            if len(Ground_Truth_dict.keys()) > len(Detection_dict.keys()):
                cal_dict = Detection_dict
            else:
                cal_dict = Ground_Truth_dict

            all_IOU = []
            for key_image_path, value_DT in cal_dict.items():
                # 这里我们根据文件名，获取GT中的字典信息：detections_boxes,category_ids
                save_name = key_image_path  # 图片名
                DT_info = Detection_dict[save_name]
                GT_info = Ground_Truth_dict[save_name]
                IOU_list = []

                for each_GT in GT_info:  # 遍历GT的每一个框
                    DT_GT_match = {}
                    GT_DT_key_and_bbox = []
                    current_GT_cate = each_GT[0]  # 当前GT的标签分类
                    current_GT_bbox = each_GT  # 当前GT的bbox
                    GT_y_min, GT_x_min, GT_y_max, GT_x_max = current_GT_bbox[1:]  # 框框的左上角点和右下角点
                    for m, each_DT in enumerate(DT_info):  # 遍历DT的每一个框

                        current_DT_bbox = each_DT  # 当前DT
                        # current_DT_cate = each_DT[0]  # 当前DT预测分类
                        DT_y_min, DT_x_min, DT_y_max, DT_x_max = current_DT_bbox[1:-1]
                        xx1 = np.max([DT_x_min, GT_x_min])
                        yy1 = np.max([DT_y_min, GT_y_min])
                        xx2 = np.min([DT_x_max, GT_x_max])
                        yy2 = np.min([DT_y_max, GT_y_max])
                        # 计算两个矩形框面积
                        area1 = (DT_x_max - DT_x_min) * (DT_y_max - DT_y_min)
                        area2 = (GT_x_max - GT_x_min) * (GT_y_max - GT_y_min)
                        inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                        IOU = inter_area / (area1 + area2 - inter_area)  # IOU = 交集面积/（DT面积+GT面积-交集面积）

                        if IOU >= 0.2:  # 这个阈值可以决定是否判断为检测正确
                            DT_GT_match[m] = IOU

                    sorted_match = dict(
                        sorted(DT_GT_match.items(), key=lambda x: x[1], reverse=True))  # 对匹配到的DT进行排序，取IOU最大的那个类别索引
                    if list(sorted_match.keys()):
                        most_matched_DT_index = list(sorted_match.items())
                        GT_DT_key_and_bbox.append(current_GT_cate)
                        for i in DT_info[most_matched_DT_index[0][0]]:
                            GT_DT_key_and_bbox.append(i)
                        GT_DT_key_and_bbox.append(round(most_matched_DT_index[0][1], 2))
                        IOU_list.append(GT_DT_key_and_bbox)
                    else:
                        GT_DT_key_and_bbox.append(current_GT_cate)
                        IOU_list.append(GT_DT_key_and_bbox)
                        all_IOU.append(IOU_list)
                new_DT_info = deepcopy(DT_info)
                # 计算过杀的DT
                for each_IOU in IOU_list:
                    for each_DT in new_DT_info:  # 遍历DT
                        # 遍历匹配的IOU列表
                        bboxes_IOU = each_IOU[2:-1]
                        bboxes_DT = each_DT[1:]
                        if len(bboxes_IOU):
                            DT_y_min, DT_x_min, DT_y_max, DT_x_max = bboxes_DT[:-1]
                            DT_y_min_IOU, DT_x_min_IOU, DT_y_max_IOU, DT_x_max_IOU = bboxes_IOU[:-1]
                            xx1 = np.max([DT_x_min, DT_x_min_IOU])
                            yy1 = np.max([DT_y_min, DT_y_min_IOU])
                            xx2 = np.min([DT_x_max, DT_x_max_IOU])
                            yy2 = np.min([DT_y_max, DT_y_max_IOU])
                            # 计算两个矩形框面积
                            area1 = (DT_x_max - DT_x_min) * (DT_y_max - DT_y_min)
                            area2 = (DT_x_max_IOU - DT_x_min_IOU) * (DT_y_max_IOU - DT_y_min_IOU)
                            inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
                            IOU = inter_area / (area1 + area2 - inter_area)  # IOU = 交集面积/（DT面积+GT面积-交集面积）
                            if IOU >= 0.9:  # 这个阈值可以决定是否判断为检测正确
                                new_DT_info.remove(each_DT)

                for i, each_DT in enumerate(new_DT_info):
                    category = self.category_id[each_DT[0]]
                    chaos_result[int(each_DT[0]) - 1][6] += 1  # 该类下过检+1
                    img_path_dict[category]["overfit"].append(save_name)  # 将该图片路径存入json
                    Overfit_count += 1

                for i, each_bboxes in enumerate(IOU_list):
                    # =================如果GT对应的DT不存在（漏检），输出黄色框并标记miss=================
                    if len(each_bboxes) == 1:
                        category = self.category_id[each_bboxes[0]]
                        chaos_result[each_bboxes[0] - 1][3] += 1  # 该类下误检+1
                        img_path_dict[category]["Miss"].append(save_name)  # 将该图片路径存入json
                        Miss_count += 1
                    # =================如果DT匹配正确，则输出绿色框=================
                    elif each_bboxes[0] == each_bboxes[1]:
                        chaos_result[int(each_bboxes[0]) - 1][1] += 1  # 该类正确+1
                        chaos_matrix[int(each_bboxes[0]) - 1][int(each_bboxes[0]) - 1] += 1  # 混淆矩阵中加1
                        OK_count += 1
                    # =================如果错误，输出红色框=================
                    elif each_bboxes[0] != each_bboxes[1]:
                        category = self.category_id[int(each_bboxes[0])]
                        chaos_result[each_bboxes[0] - 1][2] += 1  # 该类下误检+1
                        chaos_matrix[each_bboxes[0] - 1][each_bboxes[1] - 1] += 1  # 混淆矩阵中加1
                        img_path_dict[category]["Error"].append(save_name)
                        Error_count += 1

            ##########计算每一类别的AP和RECALL###########
            mAP = 0
            for i in range(len(chaos_result)):
                temp = (chaos_result[i][1] + chaos_result[i][2] + chaos_result[i][-1])
                AP = chaos_result[i][1] / temp if temp else 0
                mAP += AP
                temp = (chaos_result[i][1] + chaos_result[i][2] + chaos_result[i][3])
                RE = (chaos_result[i][1] + chaos_result[i][2]) / temp if temp else 0
                chaos_result[i][4] = round(AP, 2)
                chaos_result[i][5] = round(RE, 2)
            mAP /= len(chaos_result)
            output_chaos_result_path = os.path.join(self.output_path, "chaos_result.json")
            output_chaos_matirx_path = os.path.join(self.output_path, "chaos_matirx.json")
            with open(output_chaos_result_path, 'w') as f:
                json.dump(chaos_result.tolist(), f)
            with open(output_chaos_matirx_path, 'w') as f:
                json.dump(chaos_matrix.tolist(), f)
            #########保存混淆矩阵到excel##############
            print("---------------------统计结果---------------------")
            print("通过：{},误检：{},漏检：{},过杀：{}".format(OK_count, Error_count, Miss_count, Overfit_count))

            final_res = {
                "current_step": current_step,
                "version_id": version_id,
                "detect_res": {
                    "name_list": ["标签id", "通过", "误检", "漏检", "精度", "召回", "过杀"],
                    "category": self.columns,
                    "result": filter_nan_or_inf(chaos_result).tolist()
                },
                "chaos_res": {
                    "name_list": self.columns,
                    "result": filter_nan_or_inf(chaos_matrix).tolist()
                },
                'result_precision': mAP
            }
            # print(final_res)
            return final_res


if __name__ == "__main__":
    pass
    # frozed = r"D:\trained_model\chedeng/frozen_inference_graph.pb"
    # data_dir = r"D:\dataset\fitow_iqi_data_demo\test"
    # output_dir = r"D:\dataset\fitow_iqi_data_demo\test"
    # label_list = r"D:\dataset\fitow_iqi_data_demo\test/label_list.json"
    # worker = Auto_inference_to_DT(frozed,data_dir,output_dir,0.9,None)
    # worker.inference_on_image()
    # worker.generate_chaos_matrix()
