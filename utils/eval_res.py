# -*- encoding: utf-8 -*-
import sys
import os
import pickle
import json
import numpy as np
import copy

def coco_anno_info(coco_json):
    coco_images_dict = {}
    for i in coco_json['images']:
        coco_images_dict[str(i["id"])] = i["file_name"]

    coco_categories_dict = {}
    for i in coco_json["categories"]:
        coco_categories_dict[str(i["id"])] = i["name"]

    coco_annotations_dict = {}
    for i, each_anno in enumerate(coco_json["annotations"]):
        each_image_id = str(each_anno["image_id"])
        each_category_id = each_anno["category_id"]

        each_anno_dict = copy.deepcopy(each_anno)
        each_anno_dict["area"] = round(each_anno["area"])
        each_anno_dict["category_id"] = each_category_id

        if each_anno["segmentation"]:
            if isinstance(list, each_anno["segmentation"][0]):
                each_anno_dict["segmentation"] = list(map(round, each_anno["segmentation"][0]))
            elif isinstance(float, each_anno["segmentation"][0]):
                each_anno_dict["segmentation"] = list(map(round, each_anno["segmentation"]))
            elif isinstance(int, each_anno["segmentation"][0]):
                each_anno_dict["segmentation"] = each_anno["segmentation"]
        else:
            each_anno_dict["segmentation"] = []

        each_anno_dict["bbox"] = list(map(round, each_anno["bbox"]))
        each_anno_dict['category_name'] = coco_categories_dict[str(each_category_id)]

        if each_image_id not in coco_annotations_dict.keys():
            coco_annotations_dict[each_image_id] = [each_anno_dict]
        else:
            coco_annotations_dict[each_image_id].append(each_anno_dict)

    return coco_images_dict, coco_annotations_dict, coco_categories_dict


def detection_info(detection_json, coco_images_dict, coco_categories_dict):
    detection_annotations_dict = {}
    for i, each_anno in enumerate(detection_json):
        each_anno_dict = {}
        try:
            each_image_id = list(coco_images_dict.keys())[each_anno["order_index"]]
        except Exception as e:
            continue

        each_anno_dict["id"] = None
        each_anno_dict["image_id"] = each_image_id
        each_anno_dict["category_id"] = each_anno["category_id"]

        each_anno_dict['category_name'] = coco_categories_dict[str(each_anno["category_id"])]
        x_min, y_min, x_max, y_max = each_anno["bbox"]
        each_anno_dict["segmentation"] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        each_anno_dict["bbox"] = [x_min, y_min, x_max-x_min, y_max-y_min]
        each_anno_dict["iscrowd"] = 0

        if each_image_id not in detection_annotations_dict.keys():
            each_anno_dict["id"] = 0
            detection_annotations_dict[each_image_id] = [each_anno_dict]
        else:
            each_anno_dict["id"] = len(detection_annotations_dict[each_image_id])
            detection_annotations_dict[each_image_id].append(each_anno_dict)

    return detection_annotations_dict


def output_eval_res(pkl_file_path, GT_coco_dir, out_score_thr, save_dir):

    coco_json = json.load(open(GT_coco_dir, 'rb'))
    coco_images_dict, coco_annotations_dict, coco_categories_dict = coco_anno_info(coco_json)

    detection_json = []
    with open(pkl_file_path, 'rb') as fpkl:
        detection_datas = pickle.load(fpkl)
        num_classes = len(detection_datas[0])
        assert num_classes == len(coco_categories_dict), 'model输出类别 与 json中category_id数量不等'
        if "0" not in coco_categories_dict:
            add = 1
        else:
            add = 0

        for order_index, image_info_0 in enumerate(detection_datas):
            for category_id, image_info_1 in enumerate(image_info_0):
                for each_anno in image_info_1:
                    if each_anno.tolist()[-1] < out_score_thr:
                        continue
                    each_anno_info = dict()
                    each_anno_info['order_index'] = order_index
                    each_anno_info['category_id'] = category_id + add
                    each_anno_info['bbox'] = list(map(round, each_anno.tolist()[:-1]))
                    detection_json.append(each_anno_info)

    detection_dict = detection_info(detection_json, coco_images_dict, coco_categories_dict)

    one_dump_json = {}
    for image_id, image_name in coco_images_dict.items():
        one_dump_json['image_name'] = image_name
        one_dump_json['image_id'] = int(image_id)

        coco_pictureId = image_name.rstrip('.jpg')
        one_dump_json['pictureId'] = coco_pictureId
        one_dump_json['annotations'] = [{"ground_truth": coco_annotations_dict.get(image_id, [])},
                                        {"predictions": detection_dict.get(image_id, [])}]

        with open(os.path.join(save_dir, one_dump_json["pictureId"] + ".json"), 'w') as f:
            json.dump(one_dump_json, f, ensure_ascii=False)


if __name__ == "__main__":
    pkl_file_path = '/home/yangyu/data/fitowai/result/work_dir2/eval_1.pkl'
    GT_coco_dir = '/home/yangyu/data/fitowai/coco/annotations/train2017.json'
    save_dir = '/home/yangyu/data/fitowai/result/work_dir2/eval_1/'
    output_eval_res(pkl_file_path, GT_coco_dir, 0.6, save_dir)