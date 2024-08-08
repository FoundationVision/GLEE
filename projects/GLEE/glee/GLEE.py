#!/usr/bin/env python3
# Copyright (c) 2024 ByteDance. All Rights Reserved.
"""
GLEE Training Script.
GLEE: General Object Foundation Model for Images and Videos at Scale (CVPR 2024)
https://arxiv.org/abs/2312.09158
"""


import math
from os import DirEntry
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Dict, List
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES
from .data.datasets.objects365_v2 import categories as OBJ365_CATEGORIESV2
from .data.datasets.open_image import categories as OPENIMAGE_CATEGORIES
from .data.datasets.burst_video import BURST_CATEGORIES
from .data.datasets.vis import YTVIS_CATEGORIES_2019, OVIS_CATEGORIES, YTVIS_CATEGORIES_2021, LVVIS_CATEGORIES
from .data.datasets.uvo_video import UVO_CATEGORIES
from .data.datasets.bdd100k import BDD_DET_CATEGORIES,BDD_INST_CATEGORIES,BDD_TRACK_CATEGORIES
from .data.datasets.VisualGenome import VG_name_list
from .data.datasets.tao import TAO_CATEGORIES
from .data.datasets.odinw import odinw_category_dict
import torchvision.ops as ops
import random
from PIL import Image
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n

from .models.glee_model import GLEE_Model
from .models.matcher import HungarianMatcher
from .models.criterion import SetCriterion

from detectron2.modeling.postprocessing import sem_seg_postprocess
from .utils import box_ops
from scipy.optimize import linear_sum_assignment
import os
import copy

__all__ = ["GLEE"]



@META_ARCH_REGISTRY.register()
class GLEE(nn.Module):
    """
    Implement GLEE
    """

    def __init__(self, cfg):
        super().__init__()


        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )


        self.pseudo_video = cfg.MODEL.PSEUDO_VIDEO

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.LVIS_class_names = [cat['name'] for cat in LVIS_CATEGORIES]
        self.OBJ365_class_names = [cat['name'] for cat in OBJ365_CATEGORIESV2]
        self.OPENIMAGE_class_names = [cat['name'] for cat in OPENIMAGE_CATEGORIES]
        self.VG_name_list = VG_name_list

        self.category_set = {
            'obj365': set(list(range(365))),
            'openimage': set(list(range(601))),
            'lvis': set(list(range(1203))),
            'obj365_clip': set(list(range(365))),
            'openimage_clip': set(list(range(601))),
            'lvis_clip': set(list(range(1203))),
        }
        # self.OBJ365_set = set(list(range(365)))
        # self.OPENIMAGE_set = set(list(range(601)))
        self.brust_class_names= [cat['name'] for cat in BURST_CATEGORIES] 
        uvo_calss_name = [cat['name']+', object'  for cat in UVO_CATEGORIES[:-1] ]  + [UVO_CATEGORIES[-1]['name']]
        # print(uvo_calss_name)
        coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self.dataset_name_dicts = {
            'coco':coco_class_name,
            'coco_clip':coco_class_name,
            'lvis': self.LVIS_class_names,
            'obj365': self.OBJ365_class_names ,
            'openimage': self.OPENIMAGE_class_names, 
            'lvis_clip': self.LVIS_class_names,
            'obj365_clip': self.OBJ365_class_names ,
            'openimage_clip': self.OPENIMAGE_class_names, 
            'bdd_det': [cat['name']  for cat in BDD_DET_CATEGORIES ] ,
            'bdd_inst': [cat['name']  for cat in BDD_INST_CATEGORIES ] ,
            'ytvis19':[cat['name'] for cat in YTVIS_CATEGORIES_2019],
            'image_yt19': [cat['name'] for cat in YTVIS_CATEGORIES_2019],
            'ytvis21': [cat['name'] for cat in YTVIS_CATEGORIES_2021],
            'image_yt21': [cat['name'] for cat in YTVIS_CATEGORIES_2021],
            'ovis': [cat['name'] for cat in OVIS_CATEGORIES],
            'image_o': [cat['name'] for cat in OVIS_CATEGORIES],
            'lvvis':  [cat['name'] for cat in LVVIS_CATEGORIES], 
            'image_lv': [cat['name'] for cat in LVVIS_CATEGORIES],
            'uvo_video':uvo_calss_name,
            'burst':self.brust_class_names,
            'image_bur': self.brust_class_names,
            'image_tao': [cat['name'] for cat in TAO_CATEGORIES],
            'tao_video': [cat['name'] for cat in TAO_CATEGORIES],
            'sa1b': ['object'],
            'sa1b_clip': ['object'],
            'grounding': ['object'],
            'rvos': ['object'],
            'bdd_track_box':  [cat['name']  for cat in BDD_TRACK_CATEGORIES ] ,
            'bdd_track_seg':  [cat['name']  for cat in BDD_TRACK_CATEGORIES ] ,
            'ytbvos': ['object'],
        }
        self.num_class = {
            'lvis':len(self.LVIS_class_names),
            'obj365':365,
            'openimage':601,
            'coco':80,
            'bdd_det':10,
            'bdd_inst':8,
            'ytvis19':40,
            'image_yt19':40,
            'ovis':25,
            'image_o':25,
            'ytvis21':40,
            'image_yt21':40,
            'lvvis': len(LVVIS_CATEGORIES),
            'image_lv': len(LVVIS_CATEGORIES),
            'uvo_video': 81,
            'burst':len(self.brust_class_names),
            'image_bur': len(self.brust_class_names),
            'image_tao': len(TAO_CATEGORIES),
            'tao_video': len(TAO_CATEGORIES),
        }
        for k,v in odinw_category_dict.items():
            if k == 'odinw13_Rabbits':
                self.dataset_name_dicts.update({k:['rabbits' for cat in v ]})
            elif k == 'odinw13_EgoHands':
                self.dataset_name_dicts.update({k:['hand hands' for cat in v ]})
            elif k == 'odinw13_Mushrooms':
                self.dataset_name_dicts.update({k:['mushroom ' + cat['name']for cat in v ]})
            elif k=='odinw13_Packages':
                self.dataset_name_dicts.update({k:['packages' for cat in v ]})
            else:
                self.dataset_name_dicts.update({k:[cat['name'] for cat in v ]})
            self.num_class.update({k:len(v)})

        self.video_info = {'bz':cfg.DATALOADER.DATASET_BS[-1], 'len':cfg.INPUT.SAMPLING_FRAME_NUM}

        self.glee = GLEE_Model(cfg, matcher, self.device, self.video_info, cfg.MODEL.CONTRAS_MEAN)
      
        self.cate_sampled = False
        if cfg.MODEL.MAX_CATEGORY_LEN is not None:
            self.cate_sampled = True
            self.max_category_len = cfg.MODEL.MAX_CATEGORY_LEN 

        size_divisibility = cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.glee.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_conf": class_weight})
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        weight_dict.update({"track_loss": 2.0})
        weight_dict.update({"dist_loss": 4.0})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes", "conf"]
        else:
            losses = ["labels", "masks", "conf"]
        # building criterion
        self.criterion = SetCriterion(
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
        )

        self.test_topk_per_image = 100
        self.num_queries = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
        self.instance_on = True
        self.visaul_prompt = cfg.MODEL.VISUAL_PROMPT
        self.video_window_size = cfg.MODEL.VIDEO_WINDOW_SIZE

        self.is_lsj = cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj'

        # for VIS  
        self.is_multi_cls = True
        self.apply_cls_thres  = 0.01
        self.save_path_prefix = os.path.join(cfg.OUTPUT_DIR, "Annotations")

        #for debug ensure all loss exist for each batch
        self.all_loss_nams = ['dist_loss', 'loss_bbox', 'loss_bbox_0', 'loss_bbox_1', 'loss_bbox_2', 'loss_bbox_3', 'loss_bbox_4', 'loss_bbox_5', 'loss_bbox_6', 'loss_bbox_7', 'loss_bbox_8', 'loss_bbox_dn', 'loss_bbox_dn_0', 'loss_bbox_dn_1', 'loss_bbox_dn_2', 'loss_bbox_dn_3', 'loss_bbox_dn_4', 'loss_bbox_dn_5', 'loss_bbox_dn_6', 'loss_bbox_dn_7', 'loss_bbox_dn_8', 'loss_bbox_interm', 'loss_ce', 'loss_ce_0', 'loss_ce_1', 'loss_ce_2', 'loss_ce_3', 'loss_ce_4', 'loss_ce_5', 'loss_ce_6', 'loss_ce_7', 'loss_ce_8', 'loss_ce_dn', 'loss_ce_dn_0', 'loss_ce_dn_1', 'loss_ce_dn_2', 'loss_ce_dn_3', 'loss_ce_dn_4', 'loss_ce_dn_5', 'loss_ce_dn_6', 'loss_ce_dn_7', 'loss_ce_dn_8', 'loss_ce_interm', 'loss_conf', 'loss_conf_0', 'loss_conf_1', 'loss_conf_2', 'loss_conf_3', 'loss_conf_4', 'loss_conf_5', 'loss_conf_6', 'loss_conf_7', 'loss_conf_8', 'loss_conf_interm', 'loss_dice', 'loss_dice_0', 'loss_dice_1', 'loss_dice_2', 'loss_dice_3', 'loss_dice_4', 'loss_dice_5', 'loss_dice_6', 'loss_dice_7', 'loss_dice_8', 'loss_dice_interm', 'loss_giou', 'loss_giou_0', 'loss_giou_1', 'loss_giou_2', 'loss_giou_3', 'loss_giou_4', 'loss_giou_5', 'loss_giou_6', 'loss_giou_7', 'loss_giou_8', 'loss_giou_dn', 'loss_giou_dn_0', 'loss_giou_dn_1', 'loss_giou_dn_2', 'loss_giou_dn_3', 'loss_giou_dn_4', 'loss_giou_dn_5', 'loss_giou_dn_6', 'loss_giou_dn_7', 'loss_giou_dn_8', 'loss_giou_interm', 'loss_mask', 'loss_mask_0', 'loss_mask_1', 'loss_mask_2', 'loss_mask_3', 'loss_mask_4', 'loss_mask_5', 'loss_mask_6', 'loss_mask_7', 'loss_mask_8', 'loss_mask_interm', 'track_loss']
        self.video_task_list =  ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'lvvis', 'rvos','ytbvos', 'uvo_video','burst', 'coco_clip','obj365_clip' ,'sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']


    def vg_category_name(self, targets, task, batched_inputs):

        all_pos_categories = []
        for tgt, batchinput in zip(targets,batched_inputs):
            assert len(batchinput['object_descriptions']) == len(tgt['labels'])
            all_pos_categories = all_pos_categories + batchinput['object_descriptions']

        all_pos_set = set(all_pos_categories)

        if task == 'vg': 
            # sample 200 worlds 
            dataset_neg_category_names = random.sample(self.VG_name_list, 200)
            dataset_neg_category_names = set(dataset_neg_category_names)
            
            rest_cate = dataset_neg_category_names - all_pos_set
            label_category = list(all_pos_set)
            assert self.max_category_len*2 >= len(label_category)
            sample_cate = random.sample(rest_cate, self.max_category_len*2-len(all_pos_set))  # sampled category id
            batch_name_list = label_category+sample_cate
        elif task == 'grit': 
            dataset_neg_category_names = random.sample(self.VG_name_list, 100)
            dataset_neg_category_names = set(dataset_neg_category_names)
            
            rest_cate = dataset_neg_category_names - all_pos_set
            label_category = list(all_pos_set)
            assert self.max_category_len >= len(label_category)
            sample_cate = random.sample(rest_cate, self.max_category_len-len(all_pos_set))  # sampled category id
            batch_name_list = label_category+sample_cate
        random.shuffle(batch_name_list)

        for tgt, batchinput in zip(targets,batched_inputs): # map id into 0~100 for each batch  
            # tgt_ids = tgt["labels"]
            gt_new_ids = [batch_name_list.index(l) for l in batchinput['object_descriptions']] # index of ori CateID in new_cate_idx 
            tgt['labels'] = torch.tensor(gt_new_ids).to(tgt['labels'])

        return batch_name_list, targets
 
    
    def category_name_sampling(self, targets, task):

        all_set = self.category_set[task]
        dataset_category_names = self.dataset_name_dicts[task]
        # dataset_category_names = self.OPENIMAGE_class_names if   task == 'openimage'  else self.OBJ365_class_names
        #  for each batch rather than each image
        tgt_ids = torch.cat([v["labels"] for v in targets]).tolist()

        label_category_set = set(tgt_ids)
        rest_cate = all_set - label_category_set
        label_category = list(label_category_set)
        assert self.max_category_len >= len(label_category)
        sample_cate = random.sample(rest_cate, self.max_category_len-len(label_category))  # sampled category id
        new_cate_idx = label_category+sample_cate
        batch_name_list = [dataset_category_names[i] for i in new_cate_idx]
        
        for tgt in targets:  
            # tgt_ids = tgt["labels"]
            label_list = tgt['labels'].tolist()
            gt_new_ids = [label_category.index(l) for l in label_list]  
            tgt['labels'] = torch.tensor(gt_new_ids).to(tgt['labels'])
            
        return batch_name_list, targets
    

    
    def forward(self, batched_inputs):
        
        task, prompt_list = self.get_task_name(batched_inputs)
        
        batch_name_list = None

        if self.training:
            images = self.preprocess_image(batched_inputs, task)
            if task in self.video_task_list:
                gt_instances = [x["instances"] for x in batched_inputs]
            else:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            
            targets, prompt_list = self.prepare_targets(batched_inputs, gt_instances, images, prompt_list, task)
            
            if task in ['obj365', 'openimage', 'lvis','obj365_clip', 'lvis_clip','openimage_clip']:
                batch_name_list, targets = self.category_name_sampling( targets, task)  # sample 100 categories per iter
            elif task in ['vg', 'grit']:
                batch_name_list, targets = self.vg_category_name(targets, task, batched_inputs)  # sample 200 categories per iter
            else:
                batch_name_list = self.dataset_name_dicts[task]  # usa all category list
                
            if task in self.video_task_list : # into video tasks
                if 'spatial' in prompt_list:  #into visual prompt mode
                    (outputs, mask_dict), track_loss, dist_loss  = self.glee.video_visualP(images, prompt_list, task, targets, batch_name_list)
                else:  # category/expression guided detection/segmentation
                    (outputs, mask_dict), track_loss, dist_loss  = self.glee(images, prompt_list, task, targets, batch_name_list)
                losses = self.criterion(outputs, targets, mask_dict, task)
                losses.update({"track_loss":track_loss})
                losses.update({"dist_loss":dist_loss})
            else:
                (outputs, mask_dict), track_loss, dist_loss  = self.glee(images, prompt_list, task, targets, batch_name_list)
                losses = self.criterion(outputs, targets, mask_dict, task)
                losses.update({"track_loss":track_loss})
                losses.update({"dist_loss":dist_loss})
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                
                if ('box' in k or 'giou' in k) and task == 'grit':
                    losses[k] *= 0
            this_loss_names = set(list(losses.keys()))
            for loss_name in self.all_loss_nams:
                assert loss_name in this_loss_names , "{} is not in this batch, task is {}".format(loss_name,task)

            return losses
        else:  # evaluation
            if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'lvvis', 'uvo_video','burst', 'tao_video']:
                # return self.IDOL_inference(batched_inputs, task)
                return self.MinVIS_inference(batched_inputs, task)
            elif task in ['rvos']:
                self.inference_rvos(batched_inputs, prompt_list, task)
                return 
            elif task in ['ytbvos']:
                self.inference_ytbvos(batched_inputs, prompt_list, task)
            elif task in ['omnilabel']:
                return self.omnilabel_inference(batched_inputs, task)
            else:
                # img = batched_inputs[0]['image']
                # zero_pad = torch.zeros(3,1536,1536).to(img)
                # _,H,W = img.shape
                # zero_pad[:,:H,:W] = img
                # crop_size = (H,W)
                # batched_inputs[0]['image'] = zero_pad

                images = self.preprocess_image(batched_inputs, task)
                batch_name_list = self.dataset_name_dicts[task]

                (outputs,_),_,_ = self.glee(images, prompt_list, task, batch_name_list=batch_name_list, is_train=False)

                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_box_results = outputs["pred_boxes"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                del outputs
                processed_results = []
                for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    processed_results.append({})
                    new_size = mask_pred_result.shape[-2:]
                    if True:
                        if self.is_lsj:
                            resize_ratio = image_size[0]/max(height, width)
                            crop_size =  (int(height*resize_ratio), int(width*resize_ratio))
                        else:
                            crop_size = image_size
                        # mask_pred_result = sem_seg_postprocess(
                        #     mask_pred_result, crop_size, height, width
                        # )
                        mask_pred_result = mask_pred_result[None,][:,:,:crop_size[0],:crop_size[1]]
                        mask_pred_result = F.interpolate( mask_pred_result,   size=(height,width),  mode="bilinear",  align_corners=False,      )[0]
                        mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # instance segmentation inference
                    if self.instance_on:
                        mask_box_result = mask_box_result.to(mask_pred_result)
                        # height = new_size[0]/crop_size[0]*height
                        # width = new_size[1]/crop_size[1]*width
                        if self.is_lsj:
                            mask_box_result = self.LSJ_box_postprocess(mask_box_result, new_size, crop_size, height, width)
                        else:
                            height = new_size[0]/crop_size[0]*height
                            width = new_size[1]/crop_size[1]*width
                            mask_box_result = self.box_postprocess(mask_box_result, height, width)
                        instance_r = self.instance_inference(mask_cls_result, mask_pred_result, mask_box_result, task)
                        processed_results[-1]["instances"] = instance_r
                return processed_results
 

    def prepare_targets(self, batched_inputs, targets, images, prompt_list, task):
        img_long_size = max(images.tensor.shape[-2:])  # video data set into prompt mode with a probability of 0.4
        if  np.random.rand() > 0.0 and self.visaul_prompt and task in [
            'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'bdd_track_seg', \
             'coco_clip','sa1b_clip','lvis_clip']:  # switch into visual prompt mode
            
            if task in ['ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'sa1b_clip','coco_clip','sa1b_clip','lvis_clip','bdd_track_seg']:
                all_first_frame_num_objs = [(targets_i[0].gt_ids != -1).sum() for targets_i in targets]
                all_first_frame_num_objs = torch.stack(all_first_frame_num_objs)>0
                if all_first_frame_num_objs.all(): # each clip has a valid object in first frame
                    prompt_flag = True
                    prompt_list["spatial"] = []
                else:
                    prompt_flag = False
            else:
                prompt_flag = False

        else:
            prompt_flag = False

        if task in ['ytbvos']:
            all_first_frame_num_objs = [(targets_i[0].gt_ids != -1).sum() for targets_i in targets]
            all_first_frame_num_objs = torch.stack(all_first_frame_num_objs)>0
            if  all_first_frame_num_objs.all(): # each clip has a valid object in first frame
                prompt_flag = True
                prompt_list["spatial"] = []
            else:
                prompt_flag = False
        if task in  ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', \
            'rvos', 'coco_clip','obj365_clip','sa1b_clip','lvis_clip','openimage_clip', 'ytbvos','bdd_track_box','bdd_track_seg']:
           
            video_targets  = []
            for batched_inputs_i, targets_i in zip(batched_inputs, targets):
                h_pad, w_pad = images.tensor.shape[-2:]
                new_targets = []


                if prompt_flag :  # first frame has object
                    first_frame_valid_num = (targets_i[0].gt_ids != -1).sum().item() # the valid objects in the first frame
                    assert first_frame_valid_num>0

                    # num_prompts = random.randint(1, min(first_frame_valid_num,5) ) # keep random objects
                    num_prompts = 1 # each one prompt segment one object
                    sample_idx = random.sample(list(range(0, first_frame_valid_num)), num_prompts)  #sample index for this video
                    visualP_ids = targets_i[0].gt_ids[targets_i[0].gt_ids != -1][sample_idx].to(self.device)
                
                for frame_i, targets_per_image in enumerate(targets_i):
                    targets_per_image = targets_per_image.to(self.device)
                    if  'gt_masks' not in targets_per_image._fields.keys():
                        padded_masks = None
                    else:
                        # gt_masks = targets_per_image.gt_masks.tensor
                        if isinstance(targets_per_image.gt_masks, torch.Tensor):
                            gt_masks = targets_per_image.gt_masks
                        else:
                            gt_masks = targets_per_image.gt_masks.tensor
                        padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                        padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                    gt_classes = targets_per_image.gt_classes
                    
                    image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
                    gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                    gt_boxes = torch.clamp(gt_boxes,0,1)

                    inst_ids = targets_per_image.gt_ids
                    if prompt_flag : # inst_id that not in visualP_ids will be set to -1 and be masked 
                        not_in_prompt = [ inst_ids_i not in visualP_ids for inst_ids_i in inst_ids]
                        inst_ids[not_in_prompt] = -1

                    valid_id = inst_ids!=-1  # if a object is disappeared，its gt_ids is -1
                    if 'ori_id' in targets_per_image._fields.keys():
                        ori_id = [int(oriid) for oriid in targets_per_image.ori_id]
                    else:
                        ori_id = None

                    if padded_masks is None:
                        video_targets.append(
                        {
                            "labels": gt_classes[valid_id],
                            'inst_id':inst_ids[valid_id],
                            "masks": None,
                            "boxes":gt_boxes[valid_id],
                            "ori_id":ori_id,
                        }
                        )
                    else:
                        video_targets.append(
                            {
                                "labels": gt_classes[valid_id],
                                'inst_id':inst_ids[valid_id],
                                "masks": padded_masks[valid_id],
                                "boxes":gt_boxes[valid_id],
                                "ori_id":ori_id,
                            }
                        )
                    if prompt_flag and frame_i==0:
                        prompt_list["spatial"].append(padded_masks[valid_id]) # add the first frame gt mask as visual prompt
    
            return video_targets, prompt_list
        
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        if np.random.rand() > 0.8 and self.visaul_prompt and task in ['coco', 'sa1b', 'UVO_image', 'image_yt19', 'image_yt21', 'image_o']:  # switch into visual prompt mode
            prompt_flag = True
            prompt_list["spatial"] = []
        else:
            prompt_flag = False
        for targets_per_image in targets:
            #h, w = targets_per_image.image_size   
            #image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            if task in ['obj365','bdd_det', 'bdd_track_box','openimage','vg', 'grit'] and 'gt_masks' not in targets_per_image._fields.keys():
                padded_masks = None
            else:
                if isinstance(targets_per_image.gt_masks, torch.Tensor):
                    gt_masks = targets_per_image.gt_masks
                else:
                    gt_masks = targets_per_image.gt_masks.tensor
                # pad gt
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            gt_classes = targets_per_image.gt_classes
            #gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            # generate random visual prompt and delet un-selected gt, only keep the prompt ones
            image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
            gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            gt_boxes = torch.clamp(gt_boxes,0,1)

            if prompt_flag and len(gt_classes)>0:
                # num_prompts = random.randint(1,len(gt_classes))
                num_prompts = 1 
                sample_ids = random.sample(list(range(0, len(gt_classes))), num_prompts)
                if padded_masks is not None:
                    padded_masks = padded_masks[sample_ids]
                gt_classes = gt_classes[sample_ids]
                gt_boxes = gt_boxes[sample_ids]
            else:
                if prompt_flag:
                    prompt_flag = False
                    prompt_list.pop("spatial")
                
                
            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": padded_masks,
                    "boxes":gt_boxes,
                }
            )
            if prompt_flag:
                prompt_list["spatial"].append(padded_masks)
        return new_targets, prompt_list

    def get_task_name(self, batched_inputs):
        prompt_list = {}
        if 'dataset_name' in batched_inputs[0]:
            # print([x['dataset_name'] for x in batched_inputs])
            if 'rvos' in  batched_inputs[0]['dataset_name']:
                task = 'rvos'
                prompt_list["grounding"] = []
                for x in batched_inputs:
                    prompt_list["grounding"] += x["expressions"]
            elif 'obj365' in batched_inputs[0]['dataset_name']:
                task = 'obj365'
                if self.pseudo_video and self.training:
                    task = 'obj365_clip'
            else:
                task = batched_inputs[0]['dataset_name'] # [ovis, ytvis19, ytvis21, uvo_video, bdd_det, bdd_inst]
                if task == 'UVO_image':
                    task = 'sa1b'
                    if self.pseudo_video and self.training:
                        task = 'sa1b_clip'

        elif "expressions" in batched_inputs[0]:
            task = 'grounding'
            prompt_list["grounding"] = [x["expressions"] for x in batched_inputs]
        elif 'task' in batched_inputs[0]:
            if batched_inputs[0]['task'] == 'sa1b':
                task = 'sa1b'
                if self.pseudo_video and self.training:
                    task = 'sa1b_clip'
            else:
                task = batched_inputs[0]['task']
        elif 'not_exhaustive_category_ids' in batched_inputs[0]:
            task = 'lvis'
        else:
            task = 'undefined'
         
        if task == 'coco' and self.pseudo_video and self.training:
            task = 'coco_clip'

        if 'pseudo_video' in batched_inputs[0] and  batched_inputs[0]['pseudo_video']  == True  and self.pseudo_video and self.training:
            if task in ['lvis', 'openimage']:
                task = task+'_clip'

        return task, prompt_list
         

    def preprocess_video(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images,size_divisibility=self.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs, task):
        """
        Normalize, pad and batch the input images.
        """
        if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21','lvvis', 'uvo_video', 'burst', 'rvos','ytbvos',\
            'coco_clip','obj365_clip','sa1b_clip','lvis_clip','openimage_clip','bdd_track_box','bdd_track_seg']:
            return self.preprocess_video(batched_inputs)  #[bz [frame]]
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images,size_divisibility=self.size_divisibility)
        return images

    def instance_inference(self, mask_cls, mask_pred, mask_box_result, task):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        if task == 'grounding':
            max_inst = 1
            prob = mask_cls.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(-1), max_inst, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, mask_cls.shape[1], rounding_mode='floor')
            labels = topk_indexes % mask_cls.shape[1]
            scores_per_image = scores
            labels_per_image = labels
            mask_pred = mask_pred[topk_boxes]
            mask_box_result = mask_box_result[topk_boxes]
        elif task == 'sa1b':
            max_inst = 100
            prob = mask_cls.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(-1), max_inst, dim=0)
            scores = topk_values
            topk_boxes = torch.div(topk_indexes, mask_cls.shape[1], rounding_mode='floor')
            labels = topk_indexes % mask_cls.shape[1]
            scores_per_image = scores
            labels_per_image = labels
            mask_pred = mask_pred[topk_boxes]
            mask_box_result = mask_box_result[topk_boxes]
        
        else: 
            # [Q, K]
            if task in ['lvis', 'image_tao', 'image_bur']:
                self.test_topk_per_image = 300

            scores = mask_cls.sigmoid()  # [100, 80]
            labels = torch.arange(self.num_class[task], device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.num_class[task]
            mask_pred = mask_pred[topk_indices]
            mask_box_result = mask_box_result[topk_indices]

        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        result.pred_boxes = Boxes(mask_box_result)
        

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes
    
    def LSJ_box_postprocess(self, out_bbox,  lsj_size, crop_size, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        lsj_sclae = torch.tensor([lsj_size[1], lsj_size[0], lsj_size[1], lsj_size[0]]).to(out_bbox)
        crop_scale = torch.tensor([crop_size[1], crop_size[0], crop_size[1], crop_size[0]]).to(out_bbox)
        boxes = boxes * lsj_sclae
        boxes = boxes / crop_scale
        boxes = torch.clamp(boxes,0,1)

        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes
    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()

        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices
    def MinVIS_inference(self, batched_inputs, task):
        video_len = len(batched_inputs[0]['file_names'])


        clip_length = self.video_window_size
        batch_name_list = self.dataset_name_dicts[task]

        #split long video into clips to form a batch input 
        # if video_len > clip_length:
        num_clips = math.ceil(video_len/clip_length)
        logits_list, boxes_list, embed_list, points_list, masks_list = [], [], [], [], []
        for c in range(num_clips):
            start_idx = c*clip_length
            end_idx = (c+1)*clip_length
            clip_inputs = [{'image':batched_inputs[0]['image'][start_idx:end_idx]}]
            clip_images = self.preprocess_video(clip_inputs)
            (clip_output,_),dist,loss = self.glee(clip_images, {}, task,  batch_name_list = batch_name_list, is_train= False)
            logits_list.append(clip_output['pred_logits'])
            boxes_list.append(clip_output['pred_boxes'])
            embed_list.append(clip_output['pred_track_embed'])
            masks_list.append(clip_output['pred_masks'].cpu()) #.to(self.merge_device)
        outputs = {
            'pred_logits':torch.cat(logits_list,dim=0).detach(),
            'pred_track_embed':torch.cat(embed_list,dim=0).detach(),
            'pred_masks':torch.cat(masks_list,dim=0).detach(),
            'pred_boxes': torch.cat(boxes_list,dim=0).detach(),
        }    


        # batch_name_list  = self.dataset_name_dicts[task]
        pred_logits = list(torch.unbind(outputs['pred_logits']))
        pred_masks = list(torch.unbind(outputs['pred_masks'].cpu()))
        pred_embds = list(torch.unbind(outputs['pred_track_embed']))
        pred_boxes = list(torch.unbind(outputs['pred_boxes']))
        del outputs
        out_logits = []
        out_masks = []
        out_embds = []
        out_boxes = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0].cpu())
        out_embds.append(pred_embds[0])
        out_boxes.append(pred_boxes[0])

        for i in range(1, len(pred_logits)):
            MA_embedding = torch.stack(out_embds[-3:]).mean(0)
            indices = self.match_from_embds(MA_embedding, pred_embds[i])
            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])
            out_boxes.append(pred_boxes[i][indices, :])

        mask_cls_result = sum(out_logits)/len(out_logits)

        out_logits = torch.stack(out_logits, dim=1)  # q numc -> q t numc

        mask_pred_result = torch.stack(out_masks, dim=1) # q h w -> q t h w
        mask_box_result = torch.stack(out_boxes, dim=1) # q 4 -> q t 4
        first_resize_size = (clip_images.tensor.shape[-2], clip_images.tensor.shape[-1])

        input_per_image = batched_inputs[0]
        image_size = clip_images.image_sizes[0]  # image size without padding after data augmentation

        height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
        width = input_per_image.get("width", image_size[1])
        mask_box_result = self.box_postprocess(mask_box_result, height, width)

        return self.minvis_inference_video(mask_cls_result, mask_pred_result, mask_box_result, image_size, height, width, first_resize_size, task, out_logits, batched_inputs)


    def minvis_inference_video(self, mask_cls, mask_pred, mask_box_result, img_size, output_height, output_width, first_resize_size, task, ori_logits, batched_inputs):
        if task != 'tao_video':
            if len(mask_cls) > 0:
                # keep top-k predictions
                scores = mask_cls.sigmoid()  # [300, 40]
                num_class = self.num_class[task]
                labels = torch.arange(num_class, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
                scores_per_image, topk_indices = scores.flatten(0, 1).topk(30, sorted=False)  # select 20
                labels_per_image = labels[topk_indices]
                topk_indices = topk_indices // num_class
                mask_pred = mask_pred[topk_indices.cpu()].cpu()
                mask_box_result = mask_box_result[topk_indices]
                pred_masks = F.interpolate(
                    mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
                )
                if self.is_lsj:
                    resize_ratio = img_size[0]/max(output_height, output_width)
                    crop_size =  (int(output_height*resize_ratio), int(output_width*resize_ratio))
                else:
                    crop_size = img_size
                # resize_ratio = image_size[0]/max(height, width)
                #             crop_size =  (int(height*resize_ratio), int(width*resize_ratio))
                pred_masks = pred_masks[:, :, : crop_size[0], : crop_size[1]]
                pred_masks = F.interpolate(
                    pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
                )

                masks = pred_masks > 0.

                out_scores = scores_per_image.tolist()
                out_labels = labels_per_image.tolist()
                out_masks = [m for m in masks.cpu()]

                mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
                mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

                # xyxy2 xywh
                mask_box_result = mask_box_result.cpu().long()
                out_boxes = [m for m in mask_box_result]
            else:
                out_scores = []
                out_labels = []
                out_masks = []
                out_boxes = []

            video_output = {
                "image_size": (output_height, output_width),
                "pred_scores": out_scores,
                "pred_labels": out_labels,
                "pred_masks": out_masks,
                "pred_boxes":out_boxes,
            }
        else:  # for TAO video  teta metric
            scores = mask_cls.sigmoid()  # [300, numcls]

            topk_num = 50

            num_class = self.num_class[task]

            
            scores_per_video, topk_indices = scores.max(-1)[0].topk(topk_num, sorted=False)  
            labels_per_video =  scores[topk_indices].max(-1)[1]  # [select_num]

            mask_pred = mask_pred[topk_indices.cpu()]  #[select, len, H, W]
            mask_pred = mask_pred>0

            mask_box_result = mask_box_result[topk_indices]  #[slelct_num, len, 4]
             # xyxy2 xywh
            mask_box_result[:,:,2]=mask_box_result[:,:,2]-mask_box_result[:,:,0]
            mask_box_result[:,:,3]=mask_box_result[:,:,3]-mask_box_result[:,:,1]

            ori_logits = ori_logits[topk_indices].sigmoid()    #[slelct_num, len, num_class]
            
            image_ids = batched_inputs[0]['image_ids']
            video_id = batched_inputs[0]['video_id']
            video_len = len(image_ids)
            track_ids = torch.arange(topk_num).to(scores_per_video) + topk_num*video_id


            video_results = []
            for i,image_id in enumerate(image_ids):
                
                # frame_logits = ori_logits[:,i] # [topk_num,nun_cls]
                # scores_per_frame, labels_per_frames = frame_logits.max(-1)

                frame_boxes = mask_box_result[:,i]  
                frame_masks = mask_pred[:,i]  
                mask_valid = frame_masks.flatten(1,2).sum(-1)>5

                frame_boxes = frame_boxes[mask_valid]
                frame_scores = scores_per_video[mask_valid]
                frame_labels = labels_per_video[mask_valid]
                frame_trackids = track_ids[mask_valid]

                # box nms
                boxes_before_nms = box_ops.box_cxcywh_to_xyxy(frame_boxes)
                keep_indices = ops.nms(boxes_before_nms,frame_scores,0.5)#.tolist()

                frame_boxes = frame_boxes[keep_indices]
                frame_scores = frame_scores[keep_indices]
                frame_labels = frame_labels[keep_indices]
                frame_trackids = frame_trackids[keep_indices]

                
                for box,score,label,trackid in zip(frame_boxes,frame_scores,frame_labels,frame_trackids):
                    video_results.append(
                        {
                            "image_id" : image_id,
                            "category_id" : label.item(),
                            "bbox" : box.tolist(),
                            "score" : score.item(),
                            "track_id": trackid.item(),
                            "video_id": video_id
                        }
                    )
            video_output = video_results

        return video_output
    

  

 

