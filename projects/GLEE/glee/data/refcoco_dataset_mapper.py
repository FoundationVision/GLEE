# Copyright (c) 2024 ByteDance. All Rights Reserved.
import copy
import logging

import numpy as np
import torch
import re
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from fvcore.transforms.transform import HFlipTransform

__all__ = ["RefCOCODatasetMapper"]


def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')


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



class RefCOCODatasetMapper:
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

        self.mask_on = cfg.MODEL.MASK_ON
        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.lang_guide_det = True
        self.ordinal_nums = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if 'expressions' in dataset_dict:
            
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
            
            disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False
            dataset_dict["image"], image_shape, transforms = self.transform_img(image, disable_crop=disable_crop)
            if "expressions" in dataset_dict and dataset_dict["task"] == "grounding":
                dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                # language-guided detection
                task = dataset_dict["task"] if "task" in dataset_dict else None
                if self.lang_guide_det and task == "detection":
                    dataset_dict["expressions"] = self.prompt_test_dict[dataset_dict["dataset_name"]]
                    dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_dict["dataset_name"]]
                return dataset_dict

            if "annotations" in dataset_dict:
                instances, expressions_new = self.transform_annos(dataset_dict["annotations"], transforms, image_shape, dataset_dict)
                # add "expressions" for detection data
                dataset_dict["expressions"] = expressions_new
                instances = utils.filter_empty_instances(instances)

                if len(instances) == 0:
                    return None 
                dataset_dict["instances"] = instances
            if dataset_dict["task"] == "phrase_grounding":
                dataset_dict["task"] = "detection"
            return dataset_dict
        else:  # detection 
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
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if False:#not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict
            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.mask_on:
                        anno.pop("segmentation", None)
                    anno.pop("keypoints", None)
                if 'task' in dataset_dict and dataset_dict['task']=='vg':
                    object_description_list = [ anno['object_description'] for anno in  dataset_dict["annotations"]]
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                dataset_dict["instances"],_mask = utils.filter_empty_instances(instances, return_mask=True)

                if 'task' in dataset_dict and dataset_dict['task']=='vg': # filter empty description
                    dataset_dict["object_descriptions"] = []
                    _mask = _mask.tolist()
                    assert len(_mask) == len(object_description_list)
                    for description, _m in zip(object_description_list,_mask):
                        if _m:
                            dataset_dict["object_descriptions"].append(description)
            
            return dataset_dict





    def transform_expressions(self, expressions, transforms):
        # pick one expression if there are multiple expressions
        expression = expressions[np.random.choice(len(expressions))]
        expression = clean_string(expression)
        # deal with hflip for expression
        hflip_flag = False
        for x in transforms:
            if isinstance(x, HFlipTransform):
                hflip_flag = True
                break
        if hflip_flag:
            expression = expression.replace('left', '@').replace('right', 'left').replace('@', 'right')
        return expression


    def transform_img(self, image, disable_crop=False):
        if self.crop_gen is None or disable_crop:
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
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        return image, image_shape, transforms
    

    
    def transform_annos(self, annotations_ori, transforms, image_shape, dataset_dict):
        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in annotations_ori
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        
        # language-guided detection
        task = dataset_dict["task"] if "task" in dataset_dict else None
        if self.lang_guide_det and task == "detection":
            ind_to_class = self.ind_to_class_dict[dataset_dict["dataset_name"]]
            original_box_num = len(instances)
            instances, positive_caption_length = check_for_positive_overflow(instances, ind_to_class, self.tokenizer, self.max_query_len-2)
            if len(instances) < original_box_num:
                print("WARNING: removed {} boxes due to positive caption overflow".format(original_box_num - len(instances)))
            annotations, caption, label_to_positions = convert_object_detection_to_grounding_optimized_for_od(
                instances=instances, ind_to_class=ind_to_class,
                positive_caption_length=positive_caption_length,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_query_len-2
            )
            anno = {"annotations": annotations, "caption": caption, "label_to_positions": label_to_positions}
            anno = self.prepare(anno)
            instances.positive_map = anno["positive_map"].bool() # (N, max_seq_len). N is num of objects. bool() -> 0 or 1
            expressions_new = anno["caption"] # "expressions" are shared between detection and grounding
        elif self.lang_guide_det and task == "grounding":
            instances.positive_map = torch.ones((1, 1), dtype=torch.bool) # 1 instance, 1 (pooled) token.
            expressions_new = dataset_dict["expressions"]
        elif self.lang_guide_det and task == "phrase_grounding":
            expressions_new = dataset_dict["expressions"]
            anno = {"annotations": dataset_dict["annotations"], "caption": expressions_new}
            anno = self.prepare(anno)
            instances.positive_map = anno["positive_map"].bool() # (N, max_seq_len). N is num of objects. bool() -> 0 or 1
            expressions_new = anno["caption"] # "expressions" are shared between detection and grounding
        else:
            raise ValueError("task must be detection or grounding")
        if hasattr(instances, "gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        
        return instances, expressions_new

    def has_ordinal_num(self, expressions_list):
        flag = False
        for expression in expressions_list:
            expression_low = expression.lower()
            for word in self.ordinal_nums:
                if word in expression_low:
                    flag = True
                    break
            if flag == True:
                break
        return flag

    
       
