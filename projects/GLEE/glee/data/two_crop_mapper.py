# Copyright (c) 2024 ByteDance. All Rights Reserved.
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["COCO_CLIP_DatasetMapper"]

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


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class COCO_CLIP_DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None
        
        self.same_crop = False# cfg.INPUT.PRETRAIN_SAME_CROP

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):  # only used for training
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        
        if not self.same_crop: # twice independent crop
            #### process key frame
            # dataset_dict_ref = copy.deepcopy(dataset_dict)
            image_ref = copy.deepcopy(image)

            if self.crop_gen is None:
                image_key, transforms_key = T.apply_transform_gens(self.tfm_gens, image)
            else:
                if np.random.rand() > 0.5:
                    image_key, transforms_key = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image_key, transforms_key = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                    )

            key_image_shape = image_key.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = []
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image_key.transpose(2, 0, 1))))

            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
            key_annotations = dataset_dict.pop("annotations")
            ref_annotations = copy.deepcopy(key_annotations)
            
            annos_key = [
                utils.transform_instance_annotations(obj, transforms_key, key_image_shape)
                for obj in key_annotations
                if obj.get("iscrowd", 0) == 0
            ]
            instances_key = utils.annotations_to_instances(annos_key, key_image_shape, mask_format="bitmask")
            
            
            #### process reference frame ##########

            if self.crop_gen is None:
                image_ref, transforms_ref = T.apply_transform_gens(self.tfm_gens, image_ref)
            else:
                if np.random.rand() > 0.5:
                    image_ref, transforms_ref = T.apply_transform_gens(self.tfm_gens, image_ref)
                else:
                    image_ref, transforms_ref = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image_ref
                    )

            ref_image_shape = image_ref.shape[:2]
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image_ref.transpose(2, 0, 1))))
            annos_ref = [
                utils.transform_instance_annotations(obj, transforms_ref, ref_image_shape)
                for obj in ref_annotations
                if obj.get("iscrowd", 0) == 0
            ]
            instances_ref = utils.annotations_to_instances(annos_ref, ref_image_shape, mask_format="bitmask")



            _gt_ids = list(range(1,1+len(annos_ref)))
            instances_key.gt_ids = torch.tensor(_gt_ids)
            instances_ref.gt_ids = torch.tensor(_gt_ids)
            dataset_dict["instances"] = [filter_empty_instances(instances_key),  filter_empty_instances(instances_ref)]
          
            return dataset_dict


        else:
            if self.crop_gen is None:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                if np.random.rand() > 0.5:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image, transforms = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                    )

            image_shape = image.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            # dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            key_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            ref_image = copy.deepcopy(key_image)
            dataset_dict["image"] = [key_image,ref_image]

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.mask_on:
                        anno.pop("segmentation", None)
                    anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")

                _gt_ids = list(range(1,1+len(annos)))
                instances.gt_ids = torch.tensor(_gt_ids)

                dataset_dict["instances"] = [filter_empty_instances(instances), filter_empty_instances(instances)]
            return dataset_dict