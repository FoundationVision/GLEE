# -*- coding: utf-8 -*-
# Copyright (c) 2024 ByteDance. All Rights Reserved.
"""
GLEE config.
GLEE: General Object Foundation Model for Images and Videos at Scale (CVPR 2024)
https://arxiv.org/abs/2312.09158
"""
from detectron2.config import CfgNode as CN


def add_glee_config(cfg):
    """
    Add config for GLEE.
    """
    
    cfg.FIND_UNUSED_PARAMETERS = True
    cfg.MODEL.MAX_CATEGORY_LEN = 100
    cfg.MODEL.PSEUDO_VIDEO = False
    cfg.MODEL.FREEZE_WHOLE = False
    cfg.MODEL.CONTRAS_MEAN = False
    cfg.MODEL.CROSS_TRACK = False
    cfg.MODEL.TRACK_VERSION = 'v3'
    cfg.MODEL.EARLYFUSION = True
    cfg.MODEL.VIDEO_WINDOW_SIZE = 10
    
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
    cfg.INPUT.DATASET_MAPPER_NAME = None

    cfg.DATALOADER.DATASET_RATIO = [1, 1]
    cfg.DATALOADER.USE_DIFF_BS_SIZE = True
    cfg.DATALOADER.DATASET_BS = [2, 2]
    cfg.DATALOADER.DATASET_FILTERS = [True, True]
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True
    cfg.DATALOADER.DATASET_ANN = ['image']


    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.DATALOADER.DATASET_RATIO = [1, 1]
    cfg.DATALOADER.USE_DIFF_BS_SIZE = True
    cfg.DATALOADER.DATASET_BS = [2, 2]
    cfg.DATALOADER.USE_RFS = [False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True
    cfg.DATALOADER.DATASET_ANN = ['box', 'box']

    # Allow different datasets to use different input resolutions
    cfg.INPUT.MIN_SIZE_TRAIN_MULTI = [(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), (320, 352, 392, 416, 448, 480, 512, 544, 576, 608, 640)]
    cfg.INPUT.MAX_SIZE_TRAIN_MULTI = [1333, 768]


    # MaskDINO model config
    cfg.MODEL.MaskDINO = CN()
    cfg.MODEL.MaskDINO.LEARN_TGT = False

    # loss
    cfg.MODEL.MaskDINO.PANO_BOX_LOSS = False
    cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS = False
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.

    # cost weight
    cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.COST_BOX_WEIGHT = 5.
    cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT = 2.

    # transformer config
    cfg.MODEL.MaskDINO.NHEADS = 8
    cfg.MODEL.MaskDINO.DROPOUT = 0.1
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MaskDINO.ENC_LAYERS = 0
    cfg.MODEL.MaskDINO.DEC_LAYERS = 6
    cfg.MODEL.MaskDINO.INITIAL_PRED = True
    cfg.MODEL.MaskDINO.PRE_NORM = False
    cfg.MODEL.MaskDINO.BOX_LOSS = True
    cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.MaskDINO.DN="seg"
    cfg.MODEL.MaskDINO.DN_NOISE_SCALE=0.4
    cfg.MODEL.MaskDINO.DN_NUM=100
    cfg.MODEL.MaskDINO.PRED_CONV=False

    cfg.MODEL.MaskDINO.EVAL_FLAG = 1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    #####################

    # MaskDINO inference config
    cfg.MODEL.MaskDINO.TEST = CN()
    cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = True
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = False
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE = 0.06
    # cfg.MODEL.MaskDINO.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"

    # transformer module
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75




    cfg.MODEL.DIM_PROJ = 256
    cfg.MODEL.VISUAL_PROMPT = False
    cfg.MODEL.TEXT = CN()
    cfg.MODEL.TEXT.ARCH = 'vlpencoder'
    cfg.MODEL.TEXT.NAME= 'transformer'
    cfg.MODEL.TEXT.TOKENIZER= 'clip'
    cfg.MODEL.TEXT.CONTEXT_LENGTH= 77 # 77
    cfg.MODEL.TEXT.WIDTH= 512
    cfg.MODEL.TEXT.HEADS= 8
    cfg.MODEL.TEXT.LAYERS= 12 # 6
    cfg.MODEL.TEXT.AUTOGRESSIVE= True



    cfg.MODEL.LANGUAGE_BACKBONE = CN()
    cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT = False
    cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
    cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 768
    cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = 77 # max length of the tokenized captions. 
    cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS = 1
    # cfg.MODEL.LANGUAGE_BACKBONE.UNUSED_TOKEN = 106
    # cfg.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False
    cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX = True





    cfg.MODEL.ENCODER = CN()  
    cfg.MODEL.ENCODER.NAME= 'transformer_encoder_fpn'
    cfg.MODEL.ENCODER.IGNORE_VALUE= 255
    cfg.MODEL.ENCODER.NUM_CLASSES= 133
    cfg.MODEL.ENCODER.LOSS_WEIGHT= 1.0
    cfg.MODEL.ENCODER.CONVS_DIM= 512
    cfg.MODEL.ENCODER.MASK_DIM= 512
    cfg.MODEL.ENCODER.NORM= "GN"
    cfg.MODEL.ENCODER.IN_FEATURES= ["res2", "res3", "res4", "res5"]
    cfg.MODEL.ENCODER.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES= ["res3", "res4", "res5"]
    cfg.MODEL.ENCODER.COMMON_STRIDE= 4
    cfg.MODEL.ENCODER.TRANSFORMER_ENC_LAYERS= 6

    cfg.MODEL.DECODER = CN()  
    cfg.MODEL.DECODER.TRANSFORMER_IN_FEATURE= "multi_scale_pixel_decoder"
    cfg.MODEL.DECODER.MASK  = True
    # DETECTION= False
    # SPATIAL=
    #   ENABLED= True
    # GROUNDING=
    #   ENABLED= False
    #   MAX_LEN= 5
    #   TEXT_WEIGHT= 2.0
    #   CLASS_WEIGHT= 0.5
    # VISUAL=
    #   ENABLED= False
    # AUDIO=
    #   ENABLED= False
    # OPENIMAGE=
    #   ENABLED= False
    #   NEGATIVE_SAMPLES= 5
    #   GROUNDING=
    #     ENABLED= False
    #     MAX_LEN= 5
    # CAPTION=
    #   ENABLED= False
    #   PHRASE_PROB= 0.5
    #   SIM_THRES= 0.95
    cfg.MODEL.DECODER.HIDDEN_DIM= 512
    cfg.MODEL.DECODER.NUM_OBJECT_QUERIES= 101
    cfg.MODEL.DECODER.NHEADS= 8
    cfg.MODEL.DECODER.DROPOUT= 0.0
    cfg.MODEL.DECODER.DIM_FEEDFORWARD= 2048
    cfg.MODEL.DECODER.MAX_SPATIAL_LEN= [512, 512, 512, 512]
    cfg.MODEL.DECODER.PRE_NORM= False
    cfg.MODEL.DECODER.ENFORCE_INPUT_PROJ= False
    cfg.MODEL.DECODER.SIZE_DIVISIBILITY= 32
    cfg.MODEL.DECODER.TRAIN_NUM_POINTS= 12544
    cfg.MODEL.DECODER.OVERSAMPLE_RATIO= 3.0
    cfg.MODEL.DECODER.IMPORTANCE_SAMPLE_RATIO= 0.75
    cfg.MODEL.DECODER.DEC_LAYERS= 10  # 9 decoder layers, add one for the loss on learnable query
    cfg.MODEL.DECODER.TOP_GROUNDING_LAYERS= 10
    cfg.MODEL.DECODER.TOP_CAPTION_LAYERS= 10
    cfg.MODEL.DECODER.TOP_SPATIAL_LAYERS= 10
    cfg.MODEL.DECODER.TOP_OPENIMAGE_LAYERS= 10
    # TEST=
    #   SEMANTIC_ON= True
    #   INSTANCE_ON= True
    #   PANOPTIC_ON= True
    #   OVERLAP_THRESHOLD= 0.8
    #   OBJECT_MASK_THRESHOLD= 0.4
    #   SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE= false
    #   DETECTIONS_PER_IMAGE= 100

    cfg.ATTENTION_ARCH = CN()

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.TEXTENCODER_MULTIPLIER = 1.0
    cfg.SOLVER.LR_DECAY_RATE = None
    cfg.SOLVER.LR_DECAY_RATE_NUM_LAYERS = None


    ## support Swin backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.PRETRAINED_WEIGHT = None


    # support InterImage backbone
    cfg.MODEL.INTERNIMAGE = CN()  # large as base

    #### large
    cfg.MODEL.INTERNIMAGE.PRETRAINED_WEIGHT = None
    cfg.MODEL.INTERNIMAGE.CORE_OP = "DCNv3"
    cfg.MODEL.INTERNIMAGE.CHANNELS = 160
    cfg.MODEL.INTERNIMAGE.DEPTHS = [5, 5, 22, 5]
    cfg.MODEL.INTERNIMAGE.GROUPS =[10, 20, 40, 80]
    cfg.MODEL.INTERNIMAGE.MLP_RATIO =4.
    cfg.MODEL.INTERNIMAGE.DROP_PATH_RATE =0.0
    cfg.MODEL.INTERNIMAGE.NORM_LAYER = "LN"
    cfg.MODEL.INTERNIMAGE.LAYER_SCALE = 1.0
    cfg.MODEL.INTERNIMAGE.OFFSET_SCALE = 2.0
    cfg.MODEL.INTERNIMAGE.POST_NORM = True
    cfg.MODEL.INTERNIMAGE.WITH_CP = False
    cfg.MODEL.INTERNIMAGE.OUT_IINDICES = (0, 1, 2, 3)
    cfg.MODEL.INTERNIMAGE.DW_KERNEL_SIZE = None
    cfg.MODEL.INTERNIMAGE.RES_POST_NORM = False
    cfg.MODEL.INTERNIMAGE.LEVEL2_POST_NORM = False
    cfg.MODEL.INTERNIMAGE.LEVEL2_POST_NORM_BLOCK_IDS = None
    cfg.MODEL.INTERNIMAGE.CENTER_FEATURE_SCALE = False

    ### huge
    # cfg.MODEL.INTERNIMAGE.PRETRAINED_WEIGHT = None
    # cfg.MODEL.INTERNIMAGE.CORE_OP = "DCNv3"
    # cfg.MODEL.INTERNIMAGE.CHANNELS = 320
    # cfg.MODEL.INTERNIMAGE.DEPTHS = [6, 6, 32, 6]
    # cfg.MODEL.INTERNIMAGE.GROUPS = [10, 20, 40, 80]
    # cfg.MODEL.INTERNIMAGE.MLP_RATIO =4.
    # cfg.MODEL.INTERNIMAGE.DROP_PATH_RATE = 0.5
    # cfg.MODEL.INTERNIMAGE.NORM_LAYER = "LN"
    # cfg.MODEL.INTERNIMAGE.LAYER_SCALE = None
    # cfg.MODEL.INTERNIMAGE.OFFSET_SCALE = 1.0
    # cfg.MODEL.INTERNIMAGE.POST_NORM = False
    # cfg.MODEL.INTERNIMAGE.WITH_CP = False
    # cfg.MODEL.INTERNIMAGE.OUT_IINDICES = (0, 1, 2, 3)

    # cfg.MODEL.INTERNIMAGE.DW_KERNEL_SIZE = 5
    # cfg.MODEL.INTERNIMAGE.RES_POST_NORM = True
    # cfg.MODEL.INTERNIMAGE.LEVEL2_POST_NORM = True
    # cfg.MODEL.INTERNIMAGE.LEVEL2_POST_NORM_BLOCK_IDS = [5, 11, 17, 23, 29]
    # cfg.MODEL.INTERNIMAGE.CENTER_FEATURE_SCALE = True


    # support EVA02 backbone
    cfg.MODEL.EVA02 = CN()  # large as base

    #### large
    cfg.MODEL.EVA02.PRETRAINED_WEIGHT = None
    cfg.MODEL.EVA02.IMAGE_SIZE =  1536
    cfg.MODEL.EVA02.PATCH_SIZE =  16
    cfg.MODEL.EVA02.WINDOW_SIZE =  16
    cfg.MODEL.EVA02.DMBED_DIM =1024  
    cfg.MODEL.EVA02.DEPTH =  24
    cfg.MODEL.EVA02.NUM_HEADS =  16
    cfg.MODEL.EVA02.MLP_RATIO =   4*2/3
    cfg.MODEL.EVA02.DROP_PATH_RATE =  0.3
    cfg.MODEL.EVA02.CHECKPOINT = True
    cfg.MODEL.EVA02.WINDOW_BLOCK_INDEXES =  [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22]
    
 

    # support EVA01 backbone
    cfg.MODEL.EVA01 = CN()  # large as base

    #### large
    cfg.MODEL.EVA01.PRETRAINED_WEIGHT = None

    cfg.MODEL.EVA01.BEIT_LIKE_QKV_BIAS = True
    cfg.MODEL.EVA01.BEIT_LIKE_GAMMA = False
    cfg.MODEL.EVA01.FREEZE_PATH_EMBED = True

    cfg.MODEL.EVA01.IMAGE_SIZE =  1280  # only for correct dim in pos embed
    cfg.MODEL.EVA01.PATCH_SIZE =  16
    cfg.MODEL.EVA01.WINDOW_SIZE =  16
    cfg.MODEL.EVA01.DMBED_DIM = 1408  
    cfg.MODEL.EVA01.DEPTH =  40
    cfg.MODEL.EVA01.NUM_HEADS =  16
    cfg.MODEL.EVA01.MLP_RATIO =   6144 / 1408
    cfg.MODEL.EVA01.DROP_PATH_RATE =  0.6
    cfg.MODEL.EVA01.WINDOW_BLOCK_INDEXES =  [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38]
    
 

    """
    Add config for DeepLab.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Used for `poly` learning rate schedule.
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    # Loss type, choose from `cross_entropy`, `hard_pixel_mining`.
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    # DeepLab settings
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    # Backbone new configs
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    # ResNet stem type from: `basic`, `deeplab`
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"

 