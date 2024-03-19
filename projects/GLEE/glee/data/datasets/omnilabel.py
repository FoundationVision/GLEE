# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from lvis import LVIS



"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""
logger = logging.getLogger(__name__)

__all__ = ["load_omnilabel_json", "register_omnilabel_instances"]


def register_omnilabel_instances(name, metadata, json_file, image_root, dataset_name_in_dict=None):
    """
    """
    DatasetCatalog.register(name, lambda: load_omnilabel_json(
        json_file, image_root, dataset_name = dataset_name_in_dict))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="omnilabel", **metadata
    )


def _get_omnilabel_meta():
    categories = [{'supercategory': 'object', 'id': 1, 'name': 'object'}]
    vg_categories = sorted(categories, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in vg_categories]
    meta = {"thing_classes": thing_classes}
    return meta


def load_omnilabel_json(json_file, image_root, dataset_name=None, prompt=None):

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), \
        "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in the LVIS v1 format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            record["file_name"] = os.path.join(image_root, file_name)

        record["height"] = int(img_dict["height"])
        record["width"] = int(img_dict["width"])
        image_id = record["image_id"] = img_dict["id"]

        record["pos_description_ids"] = img_dict["pos_description_ids"]
        record["neg_description_ids"] = img_dict["neg_description_ids"]
        record["category_ids"] = img_dict["category_ids"]

        
        record["task"] = dataset_name
        dataset_dicts.append(record)
    return dataset_dicts


omnilabel_object365 = json.load(open('./projects/GLEE/glee/data/datasets/omnilabel_obj365_descriptions.json','rb'))
omnilabel_coco = json.load(open('./projects/GLEE/glee/data/datasets/omnilabel_coco_descriptions.json','rb'))
omnilabel_openimages =  json.load(open('./projects/GLEE/glee/data/datasets/omnilabel_openimages_descriptions.json','rb'))
omnilabel_all =  json.load(open('./projects/GLEE/glee/data/datasets/omnilabel_all_descriptions.json','rb'))