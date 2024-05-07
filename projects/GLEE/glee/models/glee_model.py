#!/usr/bin/env python3
# Copyright (c) 2024 ByteDance. All Rights Reserved.
# GLEE Model.
# GLEE: General Object Foundation Model for Images and Videos at Scale (CVPR 2024)
# https://arxiv.org/abs/2312.09158



import torch
import torch.nn.functional as F
from torch import nn
from detectron2.modeling import  build_backbone
from .pixel_decoder.maskdino_encoder import build_pixel_decoder
from .transformer_decoder.maskdino_decoder import build_transformer_decoder
import random
from transformers import AutoTokenizer
from collections import OrderedDict
from ..modules.point_features import point_sample
from timm.models.layers import trunc_normal_
from transformers import CLIPTokenizer,CLIPTextModel
from .vos_utils import masks_to_boxes, FeatureFuser
import numpy as np 
import math


def rand_sample(x, max_len):
    if x.shape[1] <= max_len:
        return x
    else:
        rand_idx = torch.randperm(x.shape[1])[:max_len]
        return x[:,rand_idx]


def agg_lang_feat(features, mask, pool_type="average"):
    """average pooling of language features"""
    # feat: (bs, seq_len, C)
    # mask: (bs, seq_len)
    if pool_type == "average":
        embedded = features * mask.unsqueeze(-1).float() # use mask to zero out invalid token features
        aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
    elif pool_type == "max":
        out = []
        for i in range(len(features)):
            pool_feat, _ = torch.max(features[i][mask[i]], 0) # (L, C) -> (C, )
            out.append(pool_feat)
        aggregate = torch.stack(out, dim=0) # (bs, C)
    else:
        raise ValueError("pool_type should be average or max")
    return aggregate

class GLEE_Model(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    def __init__(self, cfg, matcher, device, video_info, contras_mean):
        super().__init__()
        self.cfg = cfg
        self.matcher = matcher
        self.backbone = build_backbone(cfg)
        output_channels = [v for k,v in self.backbone._out_feature_channels.items()]
        
       
        self.lang_encoder = None
        self.find_unused_params = cfg.FIND_UNUSED_PARAMETERS
        
        if cfg.MODEL.VISUAL_PROMPT:
            self.sot_fuser = FeatureFuser(output_channels[-3:], 256)
        
        self.text_encode_type = cfg.MODEL.TEXT.ARCH
        self.early_fusion = cfg.MODEL.EARLYFUSION
         
        if cfg.MODEL.TEXT.ARCH == 'clip_frozen':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/GLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/GLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
        elif cfg.MODEL.TEXT.ARCH == 'clip_unfrozen':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/GLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/GLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))
            self.text_encode_type = 'clip_frozen'
        elif cfg.MODEL.TEXT.ARCH == 'clip_teacher':
            self.tokenizer = CLIPTokenizer.from_pretrained('projects/GLEE/clip_vit_base_patch32') 
            self.tokenizer.add_special_tokens({'cls_token': self.tokenizer.eos_token})
            self.text_encoder = CLIPTextModel.from_pretrained('projects/GLEE/clip_vit_base_patch32')
            self.text_encoder_teacher = CLIPTextModel.from_pretrained('projects/GLEE/clip_vit_base_patch32')
            self.lang_encoder = None
            for p in self.text_encoder_teacher.parameters():
                p.requires_grad = False
            self.lang_projection = nn.Parameter(torch.rand(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DIM_PROJ))

        
        # self.lang_encoder = None     
        self.pixel_decoder = build_pixel_decoder(cfg, self.backbone.output_shape())
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.predictor = build_transformer_decoder(cfg, transformer_predictor_in_channels, lang_encoder = self.lang_encoder, mask_classification=True,)
        self.to(device)
        
        self.video_info = video_info
        self.contras_mean = contras_mean

        self.track_loss_version = cfg.MODEL.TRACK_VERSION

        self.no_mask_tasks = ['obj365', 'obj365_clip','openimage', 'openimage_clip', 'vg', 'grit', 'bdd_det', 'bdd_track_box'] 


        # for visual prompt
        hidden_dim = 256
        self.max_spatial_len = [512,512,512,512]
        self.mask_sptial_embed = nn.ParameterList([nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(4)])
        trunc_normal_(self.mask_sptial_embed[0], std=.02)
        trunc_normal_(self.mask_sptial_embed[1], std=.02)
        trunc_normal_(self.mask_sptial_embed[2], std=.02)
        trunc_normal_(self.mask_sptial_embed[3], std=.02)
        # learnable positive negative indicator
        self.pn_indicator = nn.Embedding(2, hidden_dim)

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, images, prompts, task, targets=None, batch_name_list=None, is_train = True, visual_prompt_type='scribble'):
        extra =  {}
        # dist_loss = None
        early_semantic = None 
        if self.text_encode_type == 'clip_frozen':
            if task not in ['grounding','rvos']:
                assert batch_name_list
                calsses_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(calsses_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")
                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']
                token_x = token_x @ self.lang_projection
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs, 768)
                extra['class_embeddings'] = lang_feat_pool
                dist_loss =  (lang_feat_pool*0).sum()
                if self.early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 
        elif self.text_encode_type in ['EVA02_L_CLIP_frozen', 'EVA02_L_CLIP_unfreeze', 'EVA02_L_CLIP_teacher','EVA01_G_CLIP_frozen'  , 'EVA01_G_CLIP_unfreeze'  ,'EVA01_G_CLIP_teacher']:
            if task not in ['grounding','rvos']:
                assert batch_name_list
                text = self.tokenizer(batch_name_list).to('cuda')   
                lang_feat_final = self.text_encoder(text)
                lang_feat_final = lang_feat_final @ self.lang_projection
                extra['class_embeddings'] = lang_feat_final
                dist_loss =  (lang_feat_final*0).sum()
                if self.early_fusion: # class early_fusion
                    gather_all_classtoken = lang_feat_final.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 
        elif self.text_encode_type == "clip_teacher":
            if task not in ['grounding','rvos']:
                assert batch_name_list
                calsses_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(calsses_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")

                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']

                valid_mask = tokenized['attention_mask'].bool()
                token_x_teacher = self.text_encoder_teacher(*texts)['last_hidden_state']
                # if is_train:
                dist_loss =  F.mse_loss(token_x[valid_mask], token_x_teacher[valid_mask] )
                    # F.l2_loss(token_x[valid_mask], token_x_teacher[valid_mask] )  
                token_x = token_x @ self.lang_projection
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs,  768)
                extra['class_embeddings'] = lang_feat_pool 
                if self.early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 



        if 'grounding' in prompts:
 
            if self.text_encode_type == 'clip_frozen' or self.text_encode_type == 'clip_teacher':

                tokens = self.tokenizer(
                    prompts['grounding'], padding='max_length', truncation=True, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, return_tensors='pt'
                    )
                tokens = {key: value.to(images.device) for key, value in tokens.items()}

                texts = (tokens['input_ids'], tokens['attention_mask'])
                x = self.text_encoder(*texts)
                token_x = x['last_hidden_state']
                token_x = token_x @ self.lang_projection

                extra['grounding_tokens'] = token_x.permute(1,0,2) #[len,bz,C]

                non_zero_query_mask = tokens['attention_mask']
                lang_feat_pool = agg_lang_feat(token_x, non_zero_query_mask, pool_type="average").unsqueeze(1) # (bs, 1, 768)

                dist_loss =  (lang_feat_pool*0).sum()
                
                extra['grounding_nonzero_mask'] = ~non_zero_query_mask.bool()  # [bz,len]
                extra['grounding_class'] = lang_feat_pool.squeeze(1) #[bz,C
                # gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                # gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                # gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                # early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 
                early_semantic = {"hidden":token_x.float(),"masks":tokens['attention_mask']>0} 
        

        if isinstance(images,torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)




        if 'spatial' in prompts:
            ## setp 1,2,3
            key_images = [images.tensor[kid].unsqueeze(0)  for kid in range(len(images.tensor))]  #bz*[1,3,H,W]
            key_promptmasks = [m.unsqueeze(0) for m in prompts['spatial']] #bz*[1,1,H,W]

            if is_train:
                if np.random.rand() > 0.6:
                    #  image prompt mode
                    if np.random.rand() > 0.8: # box mode
                        prompt_mode = 'box'
                    else:
                        prompt_mode = 'point' # samole a point, extend a  [H//20,W//20] rectangle mask 
                        # Get a random pos point 
                        non_zero_pos_points = [rand_sample((m.nonzero()[:,1:]).t(), 1).t() for m in prompts['spatial']]
                        new_point2mask = []
                        _,h,w = prompts['spatial'][0].shape
                        point_h = h//40
                        point_w = w//40
                        for point,pmask in zip(non_zero_pos_points, key_promptmasks):
                            zeromask = torch.zeros_like(pmask)
                            zeromask[:,:, point[0,0]-point_h: point[0,0]+point_h , point[0,1]-point_w:point[0,1]+point_w ] = True
                            new_point2mask.append(zeromask)
                        key_promptmasks = new_point2mask


                    # update the visual prompt used by step2 self-attention 
                    new_prompts = []
                    for ori_mask, pmask in zip(prompts['spatial'], key_promptmasks):
                        zeromask = torch.zeros_like(ori_mask)
                        x1,y1,x2,y2  = masks_to_boxes(pmask[0])[0].long().tolist()  #[xyxy]
                        zeromask[:, y1:y2 , x1:x2] = True
                        new_prompts.append(zeromask)
                    prompts['spatial'] = new_prompts

                else:
                    prompt_mode = 'mask'
            else:
                prompt_mode = visual_prompt_type     
            
                   
            ref_feats, ref_masks = self.get_template(key_images, key_promptmasks, prompt_mode) 
            early_fusion = {"hidden":ref_feats,"masks":ref_masks} 
            if early_semantic is None:
                early_semantic = early_fusion
            else:
                early_semantic["hidden"] = torch.cat([early_semantic["hidden"],early_fusion["hidden"]],dim=1)
                early_semantic["masks"] = torch.cat([early_semantic["masks"],early_fusion["masks"]],dim=1)

        
        # bz = len(images)//2
        mask_features, _, multi_scale_features, zero_loss = self.pixel_decoder.forward_features(features, masks=None, early_fusion = early_semantic)
        if 'spatial' in prompts:
            pos_masks = prompts['spatial']
            # neg_masks = [~p for p in prompts['spatial']]
            neg_masks = [p&False for p in prompts['spatial']]
            
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})


            _,h,w = extra['spatial_query_pos_mask'][0].shape
            divisor = torch.tensor([h,w], device=mask_features.device)[None,]
            # Get mean pos spatial query
            non_zero_pos_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_pos_mask']]
            non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point, padding_value=-1).permute(1,0,2)  
            non_zero_pos_mask = (non_zero_pos_point.sum(dim=-1) < 0)  
            spatial_query_pos = point_sample(mask_features, non_zero_pos_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True) #[(N, C, P)
            spatial_query_pos = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_pos.transpose(1,2), ~non_zero_pos_mask)]).transpose(0,1).nan_to_num() # [1,bz,C]
            # Get mean neg spatial query
            non_zero_neg_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_neg_mask']]
            non_zero_neg_point = nn.utils.rnn.pad_sequence(non_zero_neg_point, padding_value=-1).permute(1,0,2)
            non_zero_neg_mask = (non_zero_neg_point.sum(dim=-1) < 0)
            spatial_query_neg = point_sample(mask_features, non_zero_neg_point.flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
            spatial_query_neg = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_neg.transpose(1,2), ~non_zero_neg_mask)]).transpose(0,1).nan_to_num()

            # Get layerwise spatial query
            src_spatial_queries = []
            src_spatial_maskings = []
            for i in range(len(multi_scale_features)):
                bs,dc,h,w = multi_scale_features[i].shape
                # src_mask_features = multi_scale_features[i].view(h,w,bs,dc)
                src_mask_features = multi_scale_features[i].permute(2,3,0,1)
                src_mask_features = src_mask_features @ self.mask_sptial_embed[i]

                non_zero_query_point_pos = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_pos_mask']]
                non_zero_query_point_neg = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_neg_mask']]
                non_zero_query_point = [torch.cat([x,y], dim=0) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x.device), -torch.ones(y.shape[0], device=y.device)]) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator, padding_value=0)
                non_zero_query_point = nn.utils.rnn.pad_sequence(non_zero_query_point, padding_value=-1).permute(1,0,2)
                non_zero_query_mask = (non_zero_query_point.sum(dim=-1) < 0)
                non_zero_query_point[non_zero_query_mask] = 0

                spatial_tokens = point_sample(src_mask_features.permute(2,3,0,1), non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype), align_corners=True).permute(2,0,1)
                spatial_tokens[pos_neg_indicator==1] += self.pn_indicator.weight[0:1]
                spatial_tokens[pos_neg_indicator==-1] += self.pn_indicator.weight[1:2]

                src_spatial_queries += [spatial_tokens]
                src_spatial_maskings += [non_zero_query_mask]

            extra['visual_prompt_tokens'] = src_spatial_queries #[len,bz,C]
            extra['visual_prompt_nonzero_mask'] = src_spatial_maskings  # [bz,len]
        
 
        ## ensure all params in loss caculation (for EVA02 backbone checkpointing propose to save memory)
        params_zero_loss = (self.pn_indicator.weight*0).sum()
        if zero_loss is not None:
            params_zero_loss += zero_loss
        for p in self.mask_sptial_embed:
            params_zero_loss += (p*0).sum()
        
        # zero_fuser_loss = 
        params_zero_loss += (self.predictor.coco_label_enc.weight*0).sum()  +\
        (self.predictor.obj365_label_enc.weight*0).sum() +\
        (self.predictor.vg_label_enc.weight*0).sum() +\
        (self.predictor.grounding_label_enc.weight*0).sum() +\
        (self.predictor.ytvis19_label_enc.weight*0).sum() +\
        (self.predictor.ytvis21_label_enc.weight*0).sum() +\
        (self.predictor.ovis_label_enc.weight*0).sum() +\
        (self.predictor.uvo_label_enc.weight*0).sum() +\
        (self.predictor.bdd_det.weight*0).sum() +\
        (self.predictor.bdd_inst.weight*0).sum()
        
        if hasattr(self,'sot_fuser') and not self.find_unused_params:  # for EVA02 checkpointing, when not in visual prompt mode, make a fake loss to ensure all parameters participate in loss calculation 
            fake_fuser_loss = self.sot_fuser([torch.zeros(1,256,1,1).to(zero_loss),torch.zeros(1,256,1,1).to(zero_loss),torch.zeros(1,256,1,1).to(zero_loss)])
            params_zero_loss += (fake_fuser_loss.sum())*0
        

        if task in ['vis', 'ovis', 'ytvis19' ,'ytvis21', 'uvo_video', 'burst', 'rvos','coco_clip','obj365_clip','sa1b_clip',\
            'bdd_track_seg', 'bdd_track_box', 'uvof_clip','lvis_clip','openimage_clip'] and is_train:
            video_outputs = []
            
            outputs = self.predictor(multi_scale_features, mask_features, extra=extra, task=task, masks=None, targets=targets)
            track_loss = self.get_tracking_contrastive_lossv3(outputs[0], targets, task)

            return outputs, track_loss, dist_loss+params_zero_loss
        else:
            outputs = self.predictor(multi_scale_features, mask_features, extra=extra, task=task, masks=None, targets=targets)
            fake_track_loss = (outputs[0]['pred_track_embed']*0).sum()
            return  outputs, fake_track_loss, dist_loss+params_zero_loss
 

    def video_visualP(self, images, prompts, task, targets=None, batch_name_list=None, is_train = True):
        extra =  {}
        # dist_loss = None
        early_semantic = None
        if  self.text_encode_type == 'clip_frozen':
            if task not in ['grounding','rvos']:
                assert batch_name_list
                calsses_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(calsses_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")

                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']
                token_x = token_x @ self.lang_projection
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs, 768)
                extra['class_embeddings'] = lang_feat_pool
                dist_loss =  (lang_feat_pool*0).sum()
                if self.early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 

        elif self.text_encode_type == "clip_teacher":
            if task not in ['grounding','rvos']:
                assert batch_name_list
                calsses_name_list = batch_name_list
                tokenized = self.tokenizer.batch_encode_plus(calsses_name_list,
                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, # 256
                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest", # max_length
                        return_special_tokens_mask=True,
                        return_tensors='pt',
                        truncation=True).to("cuda")

                texts = (tokenized['input_ids'], tokenized['attention_mask'])
                token_x = self.text_encoder(*texts)['last_hidden_state']

                valid_mask = tokenized['attention_mask'].bool()
                token_x_teacher = self.text_encoder_teacher(*texts)['last_hidden_state']
                # if is_train:
                dist_loss =  F.mse_loss(token_x[valid_mask], token_x_teacher[valid_mask] )
                    # F.l2_loss(token_x[valid_mask], token_x_teacher[valid_mask] )  
                token_x = token_x @ self.lang_projection
                lang_feat_pool = agg_lang_feat(token_x, tokenized['attention_mask'], pool_type="average")  # (bs,  768)
                extra['class_embeddings'] = lang_feat_pool 
                if self.early_fusion: # early_fusion
                    gather_all_classtoken = token_x.flatten(0,1)[tokenized['attention_mask'].flatten(0,1)>0]
                    gather_all_classtoken = gather_all_classtoken.unsqueeze(0).repeat(len(images),1,1) #[bs,L,C]
                    gather_all_classtoken_mask = torch.ones_like(gather_all_classtoken[:,:,0])>0  #[bs,L]
                    early_semantic = {"hidden":gather_all_classtoken.float(),"masks":gather_all_classtoken_mask} 

        if isinstance(images,torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)

        bz, nframe = self.video_info['bz'], self.video_info['len']
        key_prompt_frame = [i*nframe for i in range(bz)]
 
        ## setp 1,2,3
        key_images = [images.tensor[kid].unsqueeze(0)  for kid in key_prompt_frame]  #bz*[1,3,H,W]
        key_promptmasks = [m.unsqueeze(0) for m in prompts['spatial']] #bz*[1,1,H,W]


        if task in ['coco_clip', 'lvis_clip', 'uvo_video','sa1b_clip'] and np.random.rand() > 0.9:
            # into image prompt mode, givon a box and segment one object
            if np.random.rand() > 0.95: # box mode
                prompt_mode = 'box'
            else:
                prompt_mode = 'point'  
                # Get a random pos point 
                non_zero_pos_points = [rand_sample((m.nonzero()[:,1:]).t(), 1).t() for m in prompts['spatial']]
                new_point2mask = []
                _,h,w = prompts['spatial'][0].shape
                point_h = h//40
                point_w = w//40
                for point,pmask in zip(non_zero_pos_points, key_promptmasks):
                    zeromask = torch.zeros_like(pmask)
                    zeromask[:,:, point[0,0]-point_h: point[0,0]+point_h , point[0,1]-point_w:point[0,1]+point_w ] = True
                    new_point2mask.append(zeromask)
                key_promptmasks = new_point2mask

            new_prompts = []
            for ori_mask, pmask in zip(prompts['spatial'], key_promptmasks):
                zeromask = torch.zeros_like(ori_mask)
                x1,y1,x2,y2  = masks_to_boxes(pmask[0])[0].long().tolist()  #[xyxy]
                zeromask[:, y1:y2 , x1:x2] = True
                new_prompts.append(zeromask)
            prompts['spatial'] = new_prompts

        else:
            prompt_mode = 'mask'


        ref_feats, ref_masks = self.get_template(key_images, key_promptmasks, prompt_mode) 
        ref_feats = ref_feats[:,None,:,:].repeat(1,nframe,1,1).flatten(0,1) #[bz,L,c]
        ref_masks = ref_masks[:,None,:].repeat(1,nframe,1).flatten(0,1) #[bz,L]
        
        early_fusion = {"hidden":ref_feats,"masks":ref_masks} 
        if early_semantic is not None:
            early_fusion["hidden"] = torch.cat([early_fusion["hidden"],early_semantic["hidden"]],dim=1)
            early_fusion["masks"] = torch.cat([early_fusion["masks"],early_semantic["masks"]],dim=1)
         
        #######


        # bz = len(images)//2
        mask_features, _, multi_scale_features, zero_loss = self.pixel_decoder.forward_features(features, masks=None, early_fusion = early_fusion)

        # use the first frame masks as visual prompt for each video clip
        # ATTENION: all video clip frames are flatten into a batch, here we need to repeat each viusal prompt n_frame times to match the batch size
        
        prompt_mask_features = mask_features[key_prompt_frame]
        prompt_multi_scale_features = [featurei[key_prompt_frame,:,:,:] for featurei in multi_scale_features]

        # prompt_multi_scale_features = prompt_multi_scale_features+[prompt_mask_features] # use max resolution feature map as visual prompt

        if 'spatial' in prompts:
            pos_masks = prompts['spatial']
            # neg_masks = [~p for p in prompts['spatial']]
            neg_masks = [p&False for p in prompts['spatial']]
            
            extra.update({'spatial_query_pos_mask': pos_masks, 'spatial_query_neg_mask': neg_masks})


            _,h,w = extra['spatial_query_pos_mask'][0].shape
            divisor = torch.tensor([h,w], device=prompt_mask_features.device)[None,]
            # Get mean pos spatial query
            non_zero_pos_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_pos_mask']]
            non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point, padding_value=-1).permute(1,0,2)   
            
            non_zero_pos_mask = (non_zero_pos_point.sum(dim=-1) < 0)  
            spatial_query_pos = point_sample(prompt_mask_features, non_zero_pos_point.flip(dims=(2,)).type(prompt_mask_features.dtype), align_corners=True) #[(N, C, P)
            spatial_query_pos = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_pos.transpose(1,2), ~non_zero_pos_mask)]).transpose(0,1).nan_to_num() # [1,bz,C]
            # Get mean neg spatial query
            non_zero_neg_point = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[-1]).t() for m in extra['spatial_query_neg_mask']]
            non_zero_neg_point = nn.utils.rnn.pad_sequence(non_zero_neg_point, padding_value=-1).permute(1,0,2)
            non_zero_neg_mask = (non_zero_neg_point.sum(dim=-1) < 0)
            spatial_query_neg = point_sample(prompt_mask_features, non_zero_neg_point.flip(dims=(2,)).type(prompt_mask_features.dtype), align_corners=True)
            spatial_query_neg = torch.stack([x[m].mean(dim=0, keepdim=True) for x, m in zip(spatial_query_neg.transpose(1,2), ~non_zero_neg_mask)]).transpose(0,1).nan_to_num()

            # Get layerwise spatial query
            src_spatial_queries = []
            src_spatial_maskings = []
            for i in range(len(prompt_multi_scale_features)):
                bs,dc,h,w = prompt_multi_scale_features[i].shape
                # src_mask_features = multi_scale_features[i].view(h,w,bs,dc)
                src_mask_features = prompt_multi_scale_features[i].permute(2,3,0,1)
                src_mask_features = src_mask_features @ self.mask_sptial_embed[i]

                non_zero_query_point_pos = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_pos_mask']]
                non_zero_query_point_neg = [rand_sample((m.nonzero()[:,1:]/divisor).t(), self.max_spatial_len[i]).t() for m in extra['spatial_query_neg_mask']]
                non_zero_query_point = [torch.cat([x,y], dim=0) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x.device), -torch.ones(y.shape[0], device=y.device)]) for x,y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
                pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator, padding_value=0)
                non_zero_query_point = nn.utils.rnn.pad_sequence(non_zero_query_point, padding_value=-1).permute(1,0,2)
                non_zero_query_mask = (non_zero_query_point.sum(dim=-1) < 0)
                non_zero_query_point[non_zero_query_mask] = 0
                spatial_tokens = point_sample(src_mask_features.permute(2,3,0,1), non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype), align_corners=True).permute(2,0,1)
                spatial_tokens[pos_neg_indicator==1] += self.pn_indicator.weight[0:1]
                spatial_tokens[pos_neg_indicator==-1] += self.pn_indicator.weight[1:2]
                spatial_tokens = spatial_tokens[:,:,None,:].repeat(1,1,nframe,1).flatten(1,2)
                non_zero_query_mask = non_zero_query_mask[:,None,:].repeat(1,nframe,1).flatten(0,1)
                src_spatial_queries += [spatial_tokens]
                src_spatial_maskings += [non_zero_query_mask]

            extra['visual_prompt_tokens'] = src_spatial_queries #[len,bz,C]
            extra['visual_prompt_nonzero_mask'] = src_spatial_maskings  # [bz,len]
        if isinstance(images,torch.Tensor):
            features = self.backbone(images)
        else:
            features = self.backbone(images.tensor)
        # bz = len(images)//2

        ## ensure all params in loss caculation 
        params_zero_loss = zero_loss + (self.pn_indicator.weight*0).sum()
        for p in self.mask_sptial_embed:
            params_zero_loss += (p*0).sum()

        params_zero_loss += (self.predictor.coco_label_enc.weight*0).sum()  +\
        (self.predictor.obj365_label_enc.weight*0).sum() +\
        (self.predictor.vg_label_enc.weight*0).sum() +\
        (self.predictor.grounding_label_enc.weight*0).sum() +\
        (self.predictor.ytvis19_label_enc.weight*0).sum() +\
        (self.predictor.ytvis21_label_enc.weight*0).sum() +\
        (self.predictor.ovis_label_enc.weight*0).sum() +\
        (self.predictor.uvo_label_enc.weight*0).sum() +\
        (self.predictor.bdd_det.weight*0).sum() +\
        (self.predictor.bdd_inst.weight*0).sum()

        # 

        outputs = self.predictor(multi_scale_features, mask_features, extra=extra, task=task, masks=None, targets=targets)
        fake_track_loss = (outputs[0]['pred_track_embed']*0).sum()
        return  outputs, fake_track_loss, dist_loss + params_zero_loss


     

    def get_template(self, imgs, pad_masks, prompt_mode='scribble'):
        """img: (N, 3, H, W), mask: (N, 1, H, W), bbox: (1, 4)"""
        """get 4-channel template"""

        croped_img_with_mask = []

        for image_i, mask_i in zip( imgs, pad_masks):

            if prompt_mode in ['scribble','point']:
                image_with_mask = image_i + mask_i.to(image_i)
            else:
                image_with_mask = image_i 

            # image_with_mask = torch.cat([image_i,mask_i.to(image_i)],dim=1) #[1,3,H,W]
            box_i = masks_to_boxes(mask_i[0])  #[xyxy]
            box_i[:, 2:] = box_i[:, 2:] - box_i[:, :2] #xywh
            

            x, y, w, h = box_i[0].long().tolist()

            self.search_area_factor=2

            crop_sz = math.ceil(math.sqrt(w * h) * self.search_area_factor)
            x1 = max(0,round(x + 0.5 * w - crop_sz * 0.5))
            x2 = x1 + crop_sz
            y1 = max(0,round(y + 0.5 * h - crop_sz * 0.5))
            y2 = y1 + crop_sz

            im_crop = image_with_mask[:, :, y1:y2, x1:x2]
            # resize
            if im_crop.shape[-1] ==0 or im_crop.shape[-2] ==0 :
                im_crop = image_with_mask
            im_crop = F.interpolate(im_crop, (256,256), mode='bilinear', align_corners=False)
            croped_img_with_mask.append(im_crop)
        croped_img_with_mask = torch.cat(croped_img_with_mask,dim=0) #[bz,3,256,256]
        with torch.no_grad():
            ref_srcs = self.backbone(croped_img_with_mask.contiguous())
        ref_srcs = [v for k,v in ref_srcs.items()]
        ref_feats = self.sot_fuser(ref_srcs[1:]).float() #[bz,256,32,32]

        ref_feats = ref_feats.flatten(-2).permute(0, 2, 1) # (bs, L, C)
        ref_masks = torch.ones_like(ref_feats[:,:,0])>0  #[bs,L]
        
        return ref_feats, ref_masks

