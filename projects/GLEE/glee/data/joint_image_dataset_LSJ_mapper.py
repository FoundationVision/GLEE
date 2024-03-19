# Copyright (c) 2024 ByteDance. All Rights Reserved
import copy
import logging

import numpy as np
import torch
import re
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from fvcore.transforms.transform import HFlipTransform
from pycocotools import mask as coco_mask
import json
__all__ = ["Joint_Image_LSJDatasetMapper"]


def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train, has_crop):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        assert is_train, "Only support training augmentation"
        image_size = cfg.INPUT.IMAGE_SIZE
        min_scale = cfg.INPUT.MIN_SCALE
        max_scale = cfg.INPUT.MAX_SCALE

        augmentation = []

        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        if has_crop:
            augmentation.extend([
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
                ),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ])
        else:
            augmentation.extend([  # max_scale<1 so T.FixedSizeCrop will not crop images 
                T.ResizeScale(
                    min_scale=min_scale, max_scale=1.0, target_height=image_size, target_width=image_size
                ),
                T.FixedSizeCrop(crop_size=(image_size, image_size))
            ])

        return augmentation
    else:
        image_size = cfg.INPUT.IMAGE_SIZE
        test_time_augmentation = []
        test_time_augmentation.append(T.ResizeShortestEdge(short_edge_length=image_size, max_size=image_size))
        test_time_augmentation.append(T.FixedSizeCrop(crop_size=(image_size, image_size)))
        return test_time_augmentation


class Joint_Image_LSJDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        
        # Build augmentation
        self.tfm_gens = build_transform_gen(cfg, is_train, has_crop=True)
        self.tfm_gens_nocrop = build_transform_gen(cfg, is_train, has_crop= False)
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        # if cfg.INPUT.CROP.ENABLED and is_train:
        #     self.crop_gen = [
        #         T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
        #         T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
        #     ]
        # else:
        #     self.crop_gen = None
        self.mask_on = cfg.MODEL.MASK_ON
        # self.tfm_gens = build_transform_gen(cfg, is_train)
        # logging.getLogger(__name__).info(
        #     "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        # )

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
        if dataset_dict.get('task') == 'sa1b': # read the sa1b mask annotation which saved with images rather in annotation json
            mask_anno_json = json.load(open(dataset_dict["file_name"][:-3]+'json','rb'))
            assert len(mask_anno_json['annotations']) == len(dataset_dict['annotations'])
            for mask_anno, per_dict in zip(mask_anno_json['annotations'], dataset_dict['annotations']):
                per_dict['segmentation'] = mask_anno.get("segmentation", None)
        if 'expressions' in dataset_dict:  # refcoco data
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            
            disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False
            padding_mask = np.ones(image.shape[:2])
            if disable_crop:
                image, transforms = T.apply_transform_gens(self.tfm_gens_nocrop, image)
            else:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
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
                # instances, expressions_new = self.transform_annos(dataset_dict["annotations"], transforms, image_shape, dataset_dict)
                # add "expressions" for detection data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict["annotations"]
                    if obj.get("iscrowd", 0) == 0
                ]

                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # Need to filter empty instances first (due to augmentation)
                instances = utils.filter_empty_instances(instances)
                h, w = instances.image_size
                # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
                # if hasattr(instances, 'gt_masks'):
                #     gt_masks = instances.gt_masks
                #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                #     instances.gt_masks = gt_masks
                if len(instances) == 0:
                    return None 
                dataset_dict["instances"] = instances
            # if dataset_dict["task"] == "phrase_grounding":
            #     dataset_dict["task"] = "detection"
            return dataset_dict
        else:  # detection dataset [coco obj365 UVO eta]
            padding_mask = np.ones(image.shape[:2])
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            image_shape = image.shape[:2]  # h, w
            W_wop = image_shape[1] - np.sum(padding_mask[0, :])
            H_wop = image_shape[0] - np.sum(padding_mask[:, 0])
            image_shape_wop = (H_wop, W_wop) # without padding
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    # Let's always keep mask
                    anno.pop("keypoints", None)
                if dataset_dict.get('task') == 'vg' or dataset_dict.get('task') =='grit':
                    object_description_list = [ anno['object_description'] for anno in  dataset_dict["annotations"]]
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]

                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                # instances = utils.annotations_to_instances(annos, image_shape)
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                if 'gt_masks' in instances._fields.keys():
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # Need to filter empty instances first (due to augmentation)
                # instances = utils.filter_empty_instances(instances)
                instances,_mask = utils.filter_empty_instances(instances, return_mask=True)
                h, w = instances.image_size
                # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
                # if hasattr(instances, 'gt_masks'):
                #     gt_masks = instances.gt_masks
                #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                #     instances.gt_masks = gt_masks
                # NOTE: Here we get the size of image without padding. 
                # This is different from the original Mask2Former
                setattr(instances, "_image_size", image_shape_wop)
                dataset_dict["instances"] = instances

                if dataset_dict.get('task') == 'vg' or dataset_dict.get('task') =='grit': # filter empty description
                    dataset_dict["object_descriptions"] = []
                    _mask = _mask.tolist()
                    assert len(_mask) == len(object_description_list)
                    for description, _m in zip(object_description_list,_mask):
                        if _m:
                            dataset_dict["object_descriptions"].append(description)
            return dataset_dict





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

    
       
