# Copyright (c) 2024 ByteDance. All Rights Reserved.
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
This file contains functions to parse UVO dataset (YTVIS fromat) of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_UVO_dense_video"]



UVO_CATEGORIES = [
   {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'person'} ,
{'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bicycle'} ,
{'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'car'} ,
{'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'motorcycle'} ,
{'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'airplane'} ,
{'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'bus'} ,
{'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'train'} ,
{'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'truck'} ,
{'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'boat'} ,
{'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': 'traffic light'} ,
{'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': 'fire hydrant'} ,
{'color': [220, 220, 0], 'isthing': 1, 'id': 13, 'name': 'stop sign'} ,
{'color': [175, 116, 175], 'isthing': 1, 'id': 14, 'name': 'parking meter'} ,
{'color': [250, 0, 30], 'isthing': 1, 'id': 15, 'name': 'bench'} ,
{'color': [165, 42, 42], 'isthing': 1, 'id': 16, 'name': 'bird'} ,
{'color': [255, 77, 255], 'isthing': 1, 'id': 17, 'name': 'cat'} ,
{'color': [0, 226, 252], 'isthing': 1, 'id': 18, 'name': 'dog'} ,
{'color': [182, 182, 255], 'isthing': 1, 'id': 19, 'name': 'horse'} ,
{'color': [0, 82, 0], 'isthing': 1, 'id': 20, 'name': 'sheep'} ,
{'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'cow'} ,
{'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'elephant'} ,
{'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'bear'} ,
{'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'zebra'} ,
{'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'giraffe'} ,
{'color': [255, 179, 240], 'isthing': 1, 'id': 27, 'name': 'backpack'} ,
{'color': [0, 125, 92], 'isthing': 1, 'id': 28, 'name': 'umbrella'} ,
{'color': [209, 0, 151], 'isthing': 1, 'id': 31, 'name': 'handbag'} ,
{'color': [188, 208, 182], 'isthing': 1, 'id': 32, 'name': 'tie'} ,
{'color': [0, 220, 176], 'isthing': 1, 'id': 33, 'name': 'suitcase'} ,
{'color': [255, 99, 164], 'isthing': 1, 'id': 34, 'name': 'frisbee'} ,
{'color': [92, 0, 73], 'isthing': 1, 'id': 35, 'name': 'skis'} ,
{'color': [133, 129, 255], 'isthing': 1, 'id': 36, 'name': 'snowboard'} ,
{'color': [78, 180, 255], 'isthing': 1, 'id': 37, 'name': 'sports ball'} ,
{'color': [0, 228, 0], 'isthing': 1, 'id': 38, 'name': 'kite'} ,
{'color': [174, 255, 243], 'isthing': 1, 'id': 39, 'name': 'baseball bat'} ,
{'color': [45, 89, 255], 'isthing': 1, 'id': 40, 'name': 'baseball glove'} ,
{'color': [134, 134, 103], 'isthing': 1, 'id': 41, 'name': 'skateboard'} ,
{'color': [145, 148, 174], 'isthing': 1, 'id': 42, 'name': 'surfboard'} ,
{'color': [255, 208, 186], 'isthing': 1, 'id': 43, 'name': 'tennis racket'} ,
{'color': [197, 226, 255], 'isthing': 1, 'id': 44, 'name': 'bottle'} ,
{'color': [171, 134, 1], 'isthing': 1, 'id': 46, 'name': 'wine glass'} ,
{'color': [109, 63, 54], 'isthing': 1, 'id': 47, 'name': 'cup'} ,
{'color': [207, 138, 255], 'isthing': 1, 'id': 48, 'name': 'fork'} ,
{'color': [151, 0, 95], 'isthing': 1, 'id': 49, 'name': 'knife'} ,
{'color': [9, 80, 61], 'isthing': 1, 'id': 50, 'name': 'spoon'} ,
{'color': [84, 105, 51], 'isthing': 1, 'id': 51, 'name': 'bowl'} ,
{'color': [74, 65, 105], 'isthing': 1, 'id': 52, 'name': 'banana'} ,
{'color': [166, 196, 102], 'isthing': 1, 'id': 53, 'name': 'apple'} ,
{'color': [208, 195, 210], 'isthing': 1, 'id': 54, 'name': 'sandwich'} ,
{'color': [255, 109, 65], 'isthing': 1, 'id': 55, 'name': 'orange'} ,
{'color': [0, 143, 149], 'isthing': 1, 'id': 56, 'name': 'broccoli'} ,
{'color': [179, 0, 194], 'isthing': 1, 'id': 57, 'name': 'carrot'} ,
{'color': [209, 99, 106], 'isthing': 1, 'id': 58, 'name': 'hot dog'} ,
{'color': [5, 121, 0], 'isthing': 1, 'id': 59, 'name': 'pizza'} ,
{'color': [227, 255, 205], 'isthing': 1, 'id': 60, 'name': 'donut'} ,
{'color': [147, 186, 208], 'isthing': 1, 'id': 61, 'name': 'cake'} ,
{'color': [153, 69, 1], 'isthing': 1, 'id': 62, 'name': 'chair'} ,
{'color': [3, 95, 161], 'isthing': 1, 'id': 63, 'name': 'couch'} ,
{'color': [163, 255, 0], 'isthing': 1, 'id': 64, 'name': 'potted plant'} ,
{'color': [119, 0, 170], 'isthing': 1, 'id': 65, 'name': 'bed'} ,
{'color': [0, 182, 199], 'isthing': 1, 'id': 67, 'name': 'dining table'} ,
{'color': [0, 165, 120], 'isthing': 1, 'id': 70, 'name': 'toilet'} ,
{'color': [183, 130, 88], 'isthing': 1, 'id': 72, 'name': 'tv'} ,
{'color': [95, 32, 0], 'isthing': 1, 'id': 73, 'name': 'laptop'} ,
{'color': [130, 114, 135], 'isthing': 1, 'id': 74, 'name': 'mouse'} ,
{'color': [110, 129, 133], 'isthing': 1, 'id': 75, 'name': 'remote'} ,
{'color': [166, 74, 118], 'isthing': 1, 'id': 76, 'name': 'keyboard'} ,
{'color': [219, 142, 185], 'isthing': 1, 'id': 77, 'name': 'cell phone'} ,
{'color': [79, 210, 114], 'isthing': 1, 'id': 78, 'name': 'microwave'} ,
{'color': [178, 90, 62], 'isthing': 1, 'id': 79, 'name': 'oven'} ,
{'color': [65, 70, 15], 'isthing': 1, 'id': 80, 'name': 'toaster'} ,
{'color': [127, 167, 115], 'isthing': 1, 'id': 81, 'name': 'sink'} ,
{'color': [59, 105, 106], 'isthing': 1, 'id': 82, 'name': 'refrigerator'} ,
{'color': [142, 108, 45], 'isthing': 1, 'id': 84, 'name': 'book'} ,
{'color': [196, 172, 0], 'isthing': 1, 'id': 85, 'name': 'clock'} ,
{'color': [95, 54, 80], 'isthing': 1, 'id': 86, 'name': 'vase'} ,
{'color': [128, 76, 255], 'isthing': 1, 'id': 87, 'name': 'scissors'} ,
{'color': [201, 57, 1], 'isthing': 1, 'id': 88, 'name': 'teddy bear'} ,
{'color': [246, 0, 122], 'isthing': 1, 'id': 89, 'name': 'hair drier'} ,
{'color': [191, 162, 208], 'isthing': 1, 'id': 90, 'name': 'toothbrush'} ,
{'color': [191, 162, 208], 'isthing': 1, 'id': 91, 'name': 'object'} ,
]





def _get_uvo_dense_video_meta():
    thing_ids = [k["id"] for k in UVO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in UVO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 81, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in UVO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, dataset_name_in_dict="uvo_video"):
    from pycocotools.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ytvis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(ytvis_api.getCatIds())
        cats = ytvis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
        Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
        """
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    vid_ids = sorted(ytvis_api.vids.keys())
    # vids is a list of dicts, each looks something like:
    # {'license': 1,
    #  'flickr_url': ' ',
    #  'file_names': ['ff25f55852/00000.jpg', 'ff25f55852/00005.jpg', ..., 'ff25f55852/00175.jpg'],
    #  'height': 720,
    #  'width': 1280,
    #  'length': 36,
    #  'date_captured': '2019-04-11 00:55:41.903902',
    #  'id': 2232}
    vids = ytvis_api.loadVids(vid_ids)

    anns = [ytvis_api.vidToAnns[vid_id] for vid_id in vid_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(ytvis_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    vids_anns = list(zip(vids, anns))
    logger.info("Loaded {} videos in YTVIS format from {}".format(len(vids_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "category_id", "id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (vid_dict, anno_dict_list) in vids_anns:
        record = {}
        record["file_names"] = [os.path.join(image_root, vid_dict["file_names"][i]) for i in range(vid_dict["length"])]
        record["height"] = vid_dict["height"]
        record["width"] = vid_dict["width"]
        record["length"] = vid_dict["length"]
        video_id = record["video_id"] = vid_dict["id"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)

                if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                    continue

                bbox = _bboxes[frame_idx]
                segm = _segm[frame_idx]

                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                elif segm:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

                if id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                frame_objs.append(obj)
            video_objs.append(frame_objs)
        record["annotations"] = video_objs
        record["has_mask"] = True

        record["task"] = "vis"
        record["dataset_name"] = dataset_name_in_dict

        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


def register_UVO_dense_video(name, metadata, json_file, image_root, dataset_name_in_dict="uvo_video"):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name, dataset_name_in_dict=dataset_name_in_dict))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis", **metadata
    )


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis/instances_train_sub.json"
    image_root = "./datasets/ytvis/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)
