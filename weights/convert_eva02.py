# Copyright (c) 2024 ByteDance. All Rights Reserved
"""
Helper script to convert models trained with the main version of DETR to be used with the Detectron2 version.
"""
import json
import argparse

import numpy as np
import torch

# lang = torch.load('converted_seem_focalt.pt', map_location="cpu")

model_to_convert = torch.load('eva02_L_pt_m38m_p14to16.pt', map_location="cpu")
model_to_convert = model_to_convert["model"]
model_converted = {}
remove_list = []
for k in model_to_convert.keys():

    if '.rope' not in k:
        newk = 'net.'+k
        model_converted[newk] = model_to_convert[k].detach()
 

model_to_save = {"model": model_converted}
torch.save(model_to_save, 'converted_EVA02_m38m_psz14to16.pth')


