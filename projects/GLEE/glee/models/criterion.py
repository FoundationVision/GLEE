#!/usr/bin/env python3
# Copyright (c) 2024 ByteDance. All Rights Reserved.
# GLEE criterion.
# GLEE: General Object Foundation Model for Images and Videos at Scale (CVPR 2024)
# https://arxiv.org/abs/2312.09158
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Junfeng Wu

import logging
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from ..utils.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils import box_ops


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,dn="no",dn_losses=[], panoptic_on=False, semantic_ce_loss=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn=dn
        self.dn_losses=dn_losses
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.no_mask_tasks = ['obj365', 'obj365_clip', 'bdd_det', 'bdd_track_box','openimage','vg','grit', 'openimage_clip'] 

        # empty_weight_one = torch.ones(1 + 1)
        # empty_weight_one[-1] = self.eos_coef
        # self.register_buffer("empty_weight_one", empty_weight_one)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss
        self.num_classes={'coco':80, 'coconomask':80, 'coco_clip':80, 'obj365':100, 'obj365_clip':100, 'lvis':100, 'lvis_clip':100, 'openimage':100, \
            'openimage_clip':100, 'grit':100, 'vg':200, 'grounding':1,  'ytbvos':1, 'rvos':1, 'sa1b':1,'sa1b_clip':1, 'ytvis19':40, 'image_yt19':40, 'ytvis21':40,\
                 'image_yt21':40,'ovis':25, 'image_o':25, 'uvo_video':81, 'burst':1,'bdd_det':10, 'bdd_inst':8,'bdd_track_seg':8, 'bdd_track_box':8}    


    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def federated_loss(self,outputs, targets, indices):
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        src_scores = outputs['pred_logits']   
        target_score = torch.full(src_scores.shape[:2], 1, dtype=torch.int64, device=src_scores.device)
        target_score[idx] = (target_classes_o*0)
        target_score_onehot = torch.zeros([src_scores.shape[0], src_scores.shape[1], src_scores.shape[2]+1],
                                            dtype=src_scores.dtype, layout=src_scores.layout, device=src_scores.device)
        target_score_onehot.scatter_(2, target_score.unsqueeze(-1), 1)
        target_score_onehot = target_score_onehot[:,:,:-1]
        loss_conf = sigmoid_focal_loss(src_scores[idx], target_score_onehot[idx], 1, alpha=self.focal_alpha, gamma=2) 
        loss_federate_all = []

        pred_federate = outputs['pred_federat']
        for neg_score, ex_score, noex_score, target, indice in zip(pred_federate['neg'],pred_federate['ex'],pred_federate['noex'],targets,indices): 
            if len(indice[0])==0:
                continue
            ## neg:
            if neg_score.shape[1]==0:
                loss_neg = 0
            else:
                neg_label = torch.zeros_like(neg_score)
                loss_neg = sigmoid_focal_loss(neg_score, neg_label, 1, alpha=self.focal_alpha, gamma=2) 

            gt_ex_label = target['gt_ex_label'][indice[1]]
            valid = gt_ex_label!=-1

            if  ex_score.shape[1]==0 or len(indice[0][valid])==0:
                loss_exhuast = 0
            else:
                target_classes = torch.full(ex_score.shape[:1], ex_score.shape[1],
                                        dtype=torch.int64, device=ex_score.device)
                
                target_classes[indice[0][valid]] = gt_ex_label[valid]
                target_classes_onehot = torch.zeros([ex_score.shape[0], ex_score.shape[1]+1],  dtype=ex_score.dtype, layout=ex_score.layout, device=ex_score.device)

                target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
                target_classes_onehot = target_classes_onehot[:,:-1]
                loss_exhuast = sigmoid_focal_loss(ex_score, target_classes_onehot, 1, alpha=self.focal_alpha, gamma=2) 


            # no exhaustive
            
            gt_noex_label = target['gt_noex_label'][indice[1]]
            valid = gt_noex_label!=-1
            if  noex_score.shape[1]==0 or len(indice[0][valid])==0:
                loss_no_exhuast = 0
            else:
                target_classes = torch.full(noex_score.shape[:1], noex_score.shape[1],
                                        dtype=torch.int64, device=noex_score.device)

                target_classes[indice[0][valid]] = gt_noex_label[valid]
                target_classes_onehot = torch.zeros([noex_score.shape[0], noex_score.shape[1]+1],  dtype=noex_score.dtype, layout=noex_score.layout, device=noex_score.device)
                
                target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
                target_classes_onehot = target_classes_onehot[:,:-1]
                loss_no_exhuast = sigmoid_focal_loss(noex_score[indice[0][valid]], target_classes_onehot[indice[0][valid]], 1, alpha=self.focal_alpha, gamma=2) 
 
            loss_federate = (loss_neg+loss_exhuast+loss_no_exhuast)/3
            loss_federate_all.append(loss_federate)
        
        loss_federate_all = sum(loss_federate_all)/len(loss_federate_all)
        losses = {'loss_ce': loss_conf+loss_federate_all} 

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, task, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if task == 'burst' and 'pred_federat' in outputs:
            return self.federated_loss(outputs, targets, indices)
        assert 'pred_logits' in outputs
        num_boxes = max(num_boxes,1)
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
 
        num_classes = self.num_classes[task]
        target_classes = torch.full(src_logits.shape[:2], num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        

 
        losses = {
            'loss_ce': loss_ce +  (outputs['pred_scores']*0).sum(), 
            }

        return losses


    def loss_conf(self, outputs, targets, indices, num_boxes, task, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        assert 'pred_logits' in outputs
        num_boxes = max(num_boxes,1)
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
 
        if 'visual_P' in outputs and outputs['visual_P']:
            src_scores = outputs['pred_scores']   
            target_score = torch.full(src_scores.shape[:2], 1, dtype=torch.int64, device=src_scores.device)
            target_score[idx] = (target_classes_o*0)
            target_score_onehot = torch.zeros([src_scores.shape[0], src_scores.shape[1], src_scores.shape[2]+1],
                                                dtype=src_scores.dtype, layout=src_scores.layout, device=src_scores.device)
            target_score_onehot.scatter_(2, target_score.unsqueeze(-1), 1)
            target_score_onehot = target_score_onehot[:,:,:-1]
    
            loss_conf = sigmoid_focal_loss(src_scores, target_score_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * num_boxes
            losses = {
            'loss_conf':loss_conf + (outputs['pred_logits']*0).sum()
            }
        else:
            loss_conf = (outputs['pred_scores']*0).sum() + + (outputs['pred_logits']*0).sum()
            losses = {
            'loss_conf':loss_conf
            }

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, task):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        if num_boxes <1:
            losses = {}
            losses['loss_bbox'] = (outputs['pred_boxes']*0).sum()
            losses['loss_giou'] = (outputs['pred_boxes']*0).sum()
            return losses

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        src_boxes = torch.clamp(src_boxes,0,1)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes if num_boxes > 0 else loss_bbox.sum() 

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes if num_boxes > 0 else loss_giou.sum()
        # losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        isthing=target_labels<80
        target_boxes=target_boxes[isthing]
        src_boxes=src_boxes[isthing]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, task):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        num_masks = max(num_masks,1)
        assert "pred_masks" in outputs
        if task in self.no_mask_tasks and  targets[0]['masks'] is None:
            losses = {
            "loss_mask": (outputs["pred_masks"]*0).sum(),
            "loss_dice": (outputs["pred_masks"]*0).sum(),
            }
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self,mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar,pad_size=mask_dict['scalar'],mask_dict['pad_size']
        assert pad_size % scalar==0
        single_pad=pad_size//scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes,num_tgt,single_pad,scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, task):
        loss_map = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
            'conf': self.loss_conf,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, task)

    def forward(self, outputs, targets, mask_dict=None, task=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn is not "no" and mask_dict is not None:
            output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                exc_idx.append((output_idx, tgt_idx))
        if task in self.no_mask_tasks:
            indices = self.matcher(outputs_without_aux, targets, task, ["cls", "box"])
        else:
            indices = self.matcher(outputs_without_aux, targets, task)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = max(sum(len(t["labels"]) for t in targets),1)
        # num_masks = sum(len(t["labels"]) for t in targets)
        # assert num_masks>0
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device= outputs['pred_logits'].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, task))

        if self.dn != "no" and mask_dict is not None:
            l_dict={}
            for loss in self.dn_losses:
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx, num_masks*scalar, task))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != "no":
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            if self.dn == "seg":
                l_dict['loss_mask_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_dice_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                if task in self.no_mask_tasks:
                    indices = self.matcher(aux_outputs, targets, task, cost=["cls", "box"])
                else:
                    indices = self.matcher(aux_outputs, targets,task )
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, task)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i>=start:
                    if self.dn != "no" and mask_dict is not None:
                        out_=output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(
                                self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar, task))
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != "no":
                        l_dict = dict()
                        l_dict[f'loss_bbox_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_giou_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_ce_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        if self.dn == "seg":
                            l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                            l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        losses.update(l_dict)
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            # indices = self.matcher(interm_outputs, targets)
            if task in self.no_mask_tasks:
                indices = self.matcher(interm_outputs, targets, task, cost=["cls", "box"])
            else:
                indices = self.matcher(interm_outputs, targets, task)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_masks, task)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)