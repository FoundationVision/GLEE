# Copyright (c) 2024 ByteDance. All Rights Reserved.
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

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_UVO_image_json", "convert_to_coco_json", "register_UVO_image"]

BDD_DET_CATEGORIES = [
    {"id": 1, "name": "pedestrian"}, 
    {"id": 2, "name": "rider"}, 
    {"id": 3, "name": "car"}, 
    {"id": 4, "name": "truck"}, 
    {"id": 5, "name": "bus"}, 
    {"id": 6, "name": "train"}, 
    {"id": 7, "name": "motorcycle"}, 
    {"id": 8, "name": "bicycle"}, 
    {"id": 9, "name": "traffic light"}, 
    {"id": 10, "name": "traffic sign"}
    ]

BDD_INST_CATEGORIES = [
    {"id": 1, "name": "pedestrian"}, 
    {"id": 2, "name": "rider"}, 
    {"id": 3, "name": "car"}, 
    {"id": 4, "name": "truck"}, 
    {"id": 5, "name": "bus"}, 
    {"id": 6, "name": "train"}, 
    {"id": 7, "name": "motorcycle"}, 
    {"id": 8, "name": "bicycle"}
    ]

BDD_TRACK_CATEGORIES = BDD_INST_CATEGORIES


def _get_bdd_obj_det_meta():
    thing_ids = [k["id"] for k in BDD_DET_CATEGORIES]
    assert len(thing_ids) == 10, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BDD_DET_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_bdd_inst_seg_meta():
    thing_ids = [k["id"] for k in BDD_INST_CATEGORIES]
    assert len(thing_ids) == 8, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BDD_INST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

# for both MOT and MOTS
def _get_bdd_obj_track_meta():
    thing_ids = [k["id"] for k in BDD_TRACK_CATEGORIES]
    assert len(thing_ids) == 8, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BDD_TRACK_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret
