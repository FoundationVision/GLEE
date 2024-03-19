# Copyright (c) 2024 ByteDance. All Rights Reserved
import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)


 
odinw_category_dict = {


"odinw13_AerialDrone":     [{"id": 1, "name": "boat", "supercategory": "movable-objects"},
                            {"id": 2, "name": "car", "supercategory": "movable-objects"}, {"id": 3, "name":
                            "dock", "supercategory": "movable-objects"}, {"id": 4, "name": "jetski", "supercategory":
                            "movable-objects"}, {"id": 5, "name": "lift", "supercategory": "movable-objects"}] ,
"odinw13_Aquarium": [{"id": 1, "name": "fish", "supercategory": "creatures"}, {"id":
                    2, "name": "jellyfish", "supercategory": "creatures"}, {"id": 3, "name": "penguin",
                    "supercategory": "creatures"}, {"id": 4, "name": "puffin", "supercategory": "creatures"},
                    {"id": 5, "name": "shark", "supercategory": "creatures"}, {"id": 6, "name": "starfish",
                    "supercategory": "creatures"}, {"id": 7, "name": "stingray", "supercategory":"creatures"}],
"odinw13_Rabbits":  [{"id": 1, "name": "Cottontail-Rabbit", "supercategory": "Cottontail-Rabbit"}],
"odinw13_EgoHands": [{"id": 1, "name": "hand", "supercategory": "hands"}],
"odinw13_Mushrooms": [{"id": 1, "name": "CoW", "supercategory": "mushroom"}, {"id":
                        2, "name": "chanterelle", "supercategory": "mushroom"}],
"odinw13_Packages":  [{"id": 1, "name": "package", "supercategory": "packages"}],
"odinw13_PascalVOC":[{"id": 1, "name": "aeroplane", "supercategory": "VOC"}, {"id":
                        2, "name": "bicycle", "supercategory": "VOC"}, {"id": 3, "name": "bird", "supercategory":
                        "VOC"}, {"id": 4, "name": "boat", "supercategory": "VOC"}, {"id": 5, "name": "bottle",
                        "supercategory": "VOC"}, {"id": 6, "name": "bus", "supercategory": "VOC"}, {"id":
                        7, "name": "car", "supercategory": "VOC"}, {"id": 8, "name": "cat", "supercategory":
                        "VOC"}, {"id": 9, "name": "chair", "supercategory": "VOC"}, {"id": 10, "name":
                        "cow", "supercategory": "VOC"}, {"id": 11, "name": "diningtable", "supercategory":
                        "VOC"}, {"id": 12, "name": "dog", "supercategory": "VOC"}, {"id": 13, "name":
                        "horse", "supercategory": "VOC"}, {"id": 14, "name": "motorbike", "supercategory":
                        "VOC"}, {"id": 15, "name": "person", "supercategory": "VOC"}, {"id": 16, "name":
                        "pottedplant", "supercategory": "VOC"}, {"id": 17, "name": "sheep", "supercategory":
                        "VOC"}, {"id": 18, "name": "sofa", "supercategory": "VOC"}, {"id": 19, "name":
                        "train", "supercategory": "VOC"}, {"id": 20, "name": "tvmonitor", "supercategory":
                        "VOC"}],
"odinw13_Pistols":  [{"id": 1, "name": "pistol", "supercategory": "Guns"}],
"odinw13_Pothole":  [{"id": 1, "name": "pothole", "supercategory": "potholes"}],
"odinw13_Raccoon":  [{"id": 1, "name": "raccoon", "supercategory": "raccoons"}], 
"odinw13_Shellfish": [{"id": 1, "name": "Crab", "supercategory": "shellfish"}, {"id":
                        2, "name": "Lobster", "supercategory": "shellfish"}, {"id": 3, "name": "Shrimp",
                        "supercategory": "shellfish"}],
"odinw13_Thermal":  [{"id": 1, "name": "dog", "supercategory": "dogs-person"}, {"id":
                    2, "name": "person", "supercategory": "dogs-person"}],
"odinw13_Vehicles": [{"id": 1, "name": "Ambulance", "supercategory": "vehicles"},
                    {"id": 2, "name": "Bus", "supercategory": "vehicles"}, {"id": 3, "name": "Car",
                    "supercategory": "vehicles"}, {"id": 4, "name": "Motorcycle", "supercategory":
                    "vehicles"}, {"id": 5, "name": "Truck", "supercategory": "vehicles"}],
}


def _get_odinw_image_meta(name):
    thing_ids = [k["id"] for k in odinw_category_dict[name]]
    # assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in odinw_category_dict[name]]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret
