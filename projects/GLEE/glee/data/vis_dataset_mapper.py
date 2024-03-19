# Copyright (c) 2024 ByteDance. All Rights Reserved.
import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation_vis import build_augmentation
import re
from fvcore.transforms.transform import HFlipTransform
__all__ = ["YTVISDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """


    def __init__(self, cfg, is_train=True):

        self.is_train = is_train

        if cfg.INPUT.CROP.ENABLED and is_train:
            augs_nocrop, augs = build_augmentation(cfg, is_train)
            self.augmentations_nocrop = T.AugmentationList(augs_nocrop)
        else:
            augs = build_augmentation(cfg, is_train)
            self.augmentations_nocrop = None
        self.augmentations = T.AugmentationList(augs)

        self.image_format = cfg.INPUT.FORMAT
        self.sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        self.sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        self.sampling_interval = cfg.INPUT.SAMPLING_INTERVAL
        self.sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
        # self.num_classes            = num_classes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {self.augmentations}")
 

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        has_mask = dataset_dict['has_mask']
        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            start_interval = max(0, ref_frame-self.sampling_interval+1)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
            end_interval = min(video_length, ref_frame+self.sampling_interval )
            
            selected_idx = np.random.choice(
                np.array(list(range(start_idx, start_interval)) + list(range(end_interval, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)
        

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        
        task = dataset_dict["task"] if "task" in dataset_dict else None
        if task == "grounding":
            dataset_dict["expressions_ground"] = []
        if self.augmentations_nocrop is not None and self.is_train:
            if np.random.rand() > 0.5:
                selected_augmentations = self.augmentations_nocrop
            else:
                selected_augmentations = self.augmentations
        else:
            selected_augmentations = self.augmentations
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])
            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = selected_augmentations(aug_input)
            if task == "grounding":
                dataset_dict["expressions_ground"].append(self.transform_expressions(dataset_dict["expressions"], transforms))
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if dataset_dict['dataset_name'] == "ytbvos":
                pass
                # # for SOT and VOS, we need the box anno in the 1st frame during inference
                # # if (video_annos is None) or (not self.is_train):
                # #     continue
                # if not self.is_train:
                #     # NOTE copy() is to prevent annotations getting changed from applying augmentations
                #     _frame_annos = []
                #     for anno in video_annos[frame_idx]:
                #         _anno = {}
                #         for k, v in anno.items():
                #             _anno[k] = copy.deepcopy(v)
                #         _frame_annos.append(_anno)

                #     has_mask = dataset_dict["has_mask"]
                #     # USER: Implement additional transformations if you have other types of data
                #     annos = [
                #         utils.transform_instance_annotations(obj, transforms, image_shape)
                #         for obj in _frame_annos
                #         if obj.get("iscrowd", 0) == 0
                #     ]
            elif (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]

            # sorted_annos = [_get_dummy_anno(0) for _ in range(len(ids))]

            # for _anno in annos:
            #     idx = ids[_anno["id"]]
            #     sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in annos]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if dataset_dict['dataset_name'] == "ytbvos":
                ori_id_list = [x["ori_id"] if "ori_id" in x else None for x in annos]
                instances.ori_id = ori_id_list
            
            if has_mask:
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances = filter_empty_instances(instances)  
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            else:
                instances = filter_empty_instances(instances, by_box=True, by_mask=False)

            if "neg_category_ids" in dataset_dict.keys():
                is_ex = dataset_dict['exhaustive_category_ids']
                is_no_ex = dataset_dict['not_exhaustive_category_ids']
                is_ex_label = []
                is_no_ex_label = []
                for cat_id in instances.gt_classes.tolist():
                    if cat_id in is_ex:
                        is_ex_label.append( is_ex.index(cat_id) )
                    else:
                        is_ex_label.append( -1 )

                    if cat_id in is_no_ex:
                        is_no_ex_label.append( is_no_ex.index(cat_id) )
                    else:
                        is_no_ex_label.append( -1 )
                instances.is_ex_label = torch.tensor(is_ex_label)
                instances.is_no_ex_label = torch.tensor(is_no_ex_label)

            dataset_dict["instances"].append(instances) 
        if task == "grounding":
            dataset_dict["expressions"] = [dataset_dict["expressions_ground"][0] for i in range(len(dataset_dict["expressions_ground"]))]   
        
        return dataset_dict


    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        if self.is_train:
            expression = expressions[np.random.choice(len(expressions))]
            expression = clean_string(expression)
        else:
            if isinstance(expressions[0], list):
                # for refdavis, the json has been preprocessed
                # so "expressions": [["exp1", "exp2", ...]]
                expression = [clean_string(e) for e in expressions[0]]  # list
            else:
                # for refcoco and refytvos, the json has been preprocessed
                # so only one "expressions": ["exp1"]
                expression = expressions[0]
                expression = clean_string(expression)                   # str
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            if isinstance(expression, list):
                expression = [e.replace('left', '@').replace('right', 'left').replace('@', 'right') for e in expression]
            else:
                expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression

def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')
