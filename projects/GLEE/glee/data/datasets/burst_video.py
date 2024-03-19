# Copyright (c) 2024 ByteDance. All Rights Reserved
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

__all__ = ["load_ytvis_json", "register_burst_video"]



BURST_VIDEO_CATEGORIES = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 0, 'name': 'aerosol_can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'airplane'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 2, 'name': 'apricot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 3, 'name': 'apron'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 4, 'name': 'armchair'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 5, 'name': 'trash_can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 6, 'name': 'ashtray'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 7, 'name': 'atomizer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 8, 'name': 'ax'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 9, 'name': 'baby_buggy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 10, 'name': 'backpack'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 11, 'name': 'handbag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 12, 'name': 'suitcase'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 13, 'name': 'bagel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 14, 'name': 'bagpipe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 15, 'name': 'ball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 16, 'name': 'balloon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 17, 'name': 'banana'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 18, 'name': 'bandage'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 19, 'name': 'banner'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 20, 'name': 'barbell'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 21, 'name': 'barrow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 22, 'name': 'baseball_bat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 23, 'name': 'baseball_glove'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 24, 'name': 'basket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 25, 'name': 'basketball_hoop'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 26, 'name': 'basketball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 27, 'name': 'bath_mat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 28, 'name': 'beaker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 29, 'name': 'beanie'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 30, 'name': 'bear'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 31, 'name': 'bed'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 32, 'name': 'bedspread'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 33, 'name': 'cow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 34, 'name': 'beeper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 35, 'name': 'beer_can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 36, 'name': 'bell'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 37, 'name': 'bell_pepper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 38, 'name': 'bench'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 39, 'name': 'bib'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 40, 'name': 'bicycle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 41, 'name': 'binder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 42, 'name': 'binoculars'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 43, 'name': 'bird'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 44, 'name': 'birdfeeder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 45, 'name': 'birdcage'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 46, 'name': 'birdhouse'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 47, 'name': 'biscuit_(bread)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 48, 'name': 'black_sheep'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 49, 'name': 'blanket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 50, 'name': 'blender'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 51, 'name': 'gameboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 52, 'name': 'boat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 53, 'name': 'book'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 54, 'name': 'book_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 55, 'name': 'booklet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 56, 'name': 'bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 57, 'name': 'bottle_opener'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 58, 'name': 'bouquet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 59, 'name': 'bow_(weapon)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 60, 'name': 'bowl'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 61, 'name': 'bracelet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 62, 'name': 'briefcase'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 63, 'name': 'broom'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 64, 'name': 'bucket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 65, 'name': 'bull'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 66, 'name': 'bun'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 67, 'name': 'buoy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 68, 'name': 'bus_(vehicle)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 69, 'name': 'business_card'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 70, 'name': 'butcher_knife'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 71, 'name': 'cab_(taxi)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 72, 'name': 'cabinet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 73, 'name': 'calendar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 74, 'name': 'calf'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 75, 'name': 'camcorder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 76, 'name': 'camel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 77, 'name': 'camera'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 78, 'name': 'can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 79, 'name': 'candle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 80, 'name': 'canister'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 81, 'name': 'canoe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 82, 'name': 'canteen'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 83, 'name': 'bottle_cap'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 84, 'name': 'cape'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 85, 'name': 'car_(automobile)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 86, 'name': 'railcar_(part_of_a_train)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 87, 'name': 'car_battery'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 88, 'name': 'card'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 89, 'name': 'cardigan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 90, 'name': 'carrot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 91, 'name': 'tote_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 92, 'name': 'cart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 93, 'name': 'carton'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 94, 'name': 'cat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 95, 'name': 'cellular_telephone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 96, 'name': 'chain_mail'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 97, 'name': 'chair'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 98, 'name': 'chicken_(animal)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 99, 'name': 'crisp_(potato_chip)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 100, 'name': 'chocolate_bar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 101, 'name': 'chopping_board'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 102, 'name': 'chopstick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 103, 'name': 'cigar_box'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 104, 'name': 'cigarette'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 105, 'name': 'cigarette_case'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 106, 'name': 'clip'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 107, 'name': 'clipboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 108, 'name': 'clock'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 109, 'name': 'clothes_hamper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 110, 'name': 'clutch_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 111, 'name': 'coat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 112, 'name': 'coat_hanger'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 113, 'name': 'cock'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 114, 'name': 'coffee_filter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 115, 'name': 'coffee_table'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 116, 'name': 'coffeepot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 117, 'name': 'coin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 118, 'name': 'pacifier'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 119, 'name': 'computer_keyboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 120, 'name': 'cone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 121, 'name': 'control'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 122, 'name': 'convertible_(automobile)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 123, 'name': 'cooking_utensil'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 124, 'name': 'cooler_(for_food)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 125, 'name': 'edible_corn'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 126, 'name': 'cornet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 127, 'name': 'cowboy_hat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 128, 'name': 'crab_(animal)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 129, 'name': 'cracker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 130, 'name': 'crate'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 131, 'name': 'crow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 132, 'name': 'crumb'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 133, 'name': 'crutch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 134, 'name': 'cub_(animal)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 135, 'name': 'cube'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 136, 'name': 'cucumber'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 137, 'name': 'cup'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 138, 'name': 'cupcake'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 139, 'name': 'curtain'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 140, 'name': 'cushion'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 141, 'name': 'cutting_tool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 142, 'name': 'cylinder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 143, 'name': 'cymbal'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 144, 'name': 'deer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 145, 'name': 'desk'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 146, 'name': 'die'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 147, 'name': 'dining_table'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 148, 'name': 'dish'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 149, 'name': 'dispenser'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 150, 'name': 'dog'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 151, 'name': 'doormat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 152, 'name': 'drawer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 153, 'name': 'dress_hat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 154, 'name': 'drone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 155, 'name': 'drum_(musical_instrument)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 156, 'name': 'drumstick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 157, 'name': 'duck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 158, 'name': 'duckling'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 159, 'name': 'duffel_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 160, 'name': 'dustpan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 161, 'name': 'eagle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 162, 'name': 'earphone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 163, 'name': 'earring'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 164, 'name': 'egg'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 165, 'name': 'eggbeater'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 166, 'name': 'refrigerator'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 167, 'name': 'elephant'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 168, 'name': 'envelope'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 169, 'name': 'fan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 170, 'name': 'faucet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 171, 'name': 'Ferris_wheel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 172, 'name': 'file_(tool)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 173, 'name': 'fire_engine'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 174, 'name': 'fish'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 175, 'name': 'fishing_rod'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 176, 'name': 'flag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 177, 'name': 'flashlight'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 178, 'name': 'flute_glass'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 179, 'name': 'football_helmet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 180, 'name': 'footstool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 181, 'name': 'fork'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 182, 'name': 'frisbee'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 183, 'name': 'frog'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 184, 'name': 'frying_pan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 185, 'name': 'garbage'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 186, 'name': 'garbage_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 187, 'name': 'garden_hose'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 188, 'name': 'gasmask'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 189, 'name': 'giant_panda'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 190, 'name': 'gift_wrap'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 191, 'name': 'giraffe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 192, 'name': 'glove'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 193, 'name': 'goat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 194, 'name': 'goggles'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 195, 'name': 'golf_club'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 196, 'name': 'golfcart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 197, 'name': 'goose'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 198, 'name': 'gorilla'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 199, 'name': 'grocery_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 200, 'name': 'guitar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 201, 'name': 'gun'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 202, 'name': 'hair_spray'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 203, 'name': 'hairbrush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 204, 'name': 'hamburger'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 205, 'name': 'hammer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 206, 'name': 'hamster'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 207, 'name': 'hair_dryer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 208, 'name': 'hand_towel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 209, 'name': 'handcart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 210, 'name': 'handcuff'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 211, 'name': 'handkerchief'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 212, 'name': 'handle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 213, 'name': 'handsaw'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 214, 'name': 'harmonium'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 215, 'name': 'hat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 216, 'name': 'headscarf'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 217, 'name': 'helicopter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 218, 'name': 'helmet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 219, 'name': 'heron'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 220, 'name': 'hippopotamus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 221, 'name': 'hockey_stick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 222, 'name': 'hog'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 223, 'name': 'horse'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 224, 'name': 'hose'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 225, 'name': 'polar_bear'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 226, 'name': 'icecream'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 227, 'name': 'igniter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 228, 'name': 'iPod'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 229, 'name': 'iron_(for_clothing)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 230, 'name': 'jacket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 231, 'name': 'jean'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 232, 'name': 'jeep'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 233, 'name': 'jersey'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 234, 'name': 'kayak'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 235, 'name': 'kettle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 236, 'name': 'kite'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 237, 'name': 'knife'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 238, 'name': 'knitting_needle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 239, 'name': 'knob'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 240, 'name': 'ladle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 241, 'name': 'lamp'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 242, 'name': 'lanyard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 243, 'name': 'laptop_computer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 244, 'name': 'latch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 245, 'name': 'lawn_mower'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 246, 'name': 'lemon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 247, 'name': 'lettuce'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 248, 'name': 'life_buoy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 249, 'name': 'life_jacket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 250, 'name': 'lion'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 251, 'name': 'lizard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 252, 'name': 'log'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 253, 'name': 'magazine'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 254, 'name': 'mailbox_(at_home)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 255, 'name': 'mallet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 256, 'name': 'marker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 257, 'name': 'mat_(gym_equipment)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 258, 'name': 'matchbox'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 259, 'name': 'mattress'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 260, 'name': 'measuring_cup'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 261, 'name': 'measuring_stick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 262, 'name': 'medicine'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 263, 'name': 'microphone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 264, 'name': 'minivan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 265, 'name': 'mirror'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 266, 'name': 'mixer_(kitchen_tool)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 267, 'name': 'money'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 268, 'name': 'monitor_(computer_equipment) computer_monitor'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 269, 'name': 'monkey'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 270, 'name': 'motor_scooter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 271, 'name': 'motorboat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 272, 'name': 'motorcycle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 273, 'name': 'mouse_(animal_rodent)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 274, 'name': 'mouse_(computer_equipment)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 275, 'name': 'musical_instrument'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 276, 'name': 'napkin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 277, 'name': 'necklace'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 278, 'name': 'nest'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 279, 'name': 'newsstand'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 280, 'name': 'notebook'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 281, 'name': 'notepad'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 282, 'name': 'nut'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 283, 'name': 'oar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 284, 'name': 'onion'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 285, 'name': 'orange_(fruit)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 286, 'name': 'packet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 287, 'name': 'paddle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 288, 'name': 'paintbox'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 289, 'name': 'paintbrush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 290, 'name': 'painting'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 291, 'name': 'pajamas'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 292, 'name': 'palette'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 293, 'name': 'pan_(for_cooking)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 294, 'name': 'paper_towel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 295, 'name': 'parachute'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 296, 'name': 'parrot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 297, 'name': 'peeler_(tool_for_fruit_and_vegetables)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 298, 'name': 'pelican'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 299, 'name': 'pen'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 300, 'name': 'pencil'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 301, 'name': 'penguin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 302, 'name': 'pepper_mill'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 303, 'name': 'persimmon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 304, 'name': 'person'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 305, 'name': 'petfood'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 306, 'name': 'phonograph_record'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 307, 'name': 'piano'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 308, 'name': 'pickle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 309, 'name': 'pickup_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 310, 'name': 'pigeon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 311, 'name': 'pillow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 312, 'name': 'ping-pong_ball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 313, 'name': 'tobacco_pipe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 314, 'name': 'pipe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 315, 'name': 'pistol'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 316, 'name': 'pitcher_(vessel_for_liquid)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 317, 'name': 'pizza'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 318, 'name': 'plate'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 319, 'name': 'pliers'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 320, 'name': 'pole'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 321, 'name': 'pony'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 322, 'name': 'poster'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 323, 'name': 'pot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 324, 'name': 'flowerpot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 325, 'name': 'potato'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 326, 'name': 'pouch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 327, 'name': 'power_shovel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 328, 'name': 'pumpkin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 329, 'name': 'puppet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 330, 'name': 'rabbit'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 331, 'name': 'race_car'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 332, 'name': 'racket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 333, 'name': 'raft'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 334, 'name': 'rag_doll'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 335, 'name': 'rat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 336, 'name': 'razorblade'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 337, 'name': 'record_player'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 338, 'name': 'remote_control'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 339, 'name': 'rhinoceros'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 340, 'name': 'rifle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 341, 'name': 'rubber_band'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 342, 'name': 'runner_(carpet)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 343, 'name': 'saddle_blanket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 344, 'name': 'saltshaker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 345, 'name': 'sandal_(type_of_shoe)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 346, 'name': 'sandwich'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 347, 'name': 'saucepan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 348, 'name': 'saxophone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 349, 'name': 'scarf'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 350, 'name': 'school_bus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 351, 'name': 'scissors'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 352, 'name': 'scoreboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 353, 'name': 'scraper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 354, 'name': 'screwdriver'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 355, 'name': 'scrubbing_brush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 356, 'name': 'sculpture'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 357, 'name': 'serving_dish'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 358, 'name': 'shaker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 359, 'name': 'shampoo'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 360, 'name': 'shark'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 361, 'name': 'sharpener'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 362, 'name': 'shaver_(electric)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 363, 'name': 'shawl'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 364, 'name': 'shears'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 365, 'name': 'sheep'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 366, 'name': 'shield'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 367, 'name': 'shirt'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 368, 'name': 'shoe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 369, 'name': 'shopping_cart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 370, 'name': 'short_pants'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 371, 'name': 'shoulder_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 372, 'name': 'shovel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 373, 'name': 'sieve'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 374, 'name': 'skateboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 375, 'name': 'ski'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 376, 'name': 'ski_pole'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 377, 'name': 'skirt'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 378, 'name': 'sled'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 379, 'name': 'slipper_(footwear)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 380, 'name': 'soap'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 381, 'name': 'sock'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 382, 'name': 'sofa'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 383, 'name': 'soupspoon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 384, 'name': 'spatula'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 385, 'name': 'spectacles'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 386, 'name': 'spider'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 387, 'name': 'sponge'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 388, 'name': 'spoon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 389, 'name': 'spotlight'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 390, 'name': 'squirrel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 391, 'name': 'statue_(sculpture)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 392, 'name': 'steering_wheel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 393, 'name': 'step_stool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 394, 'name': 'stool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 395, 'name': 'stove'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 396, 'name': 'straw_(for_drinking)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 397, 'name': 'strawberry'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 398, 'name': 'street_sign'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 399, 'name': 'stylus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 400, 'name': 'sugar_bowl'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 401, 'name': 'sunglasses'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 402, 'name': 'sunscreen'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 403, 'name': 'surfboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 404, 'name': 'mop'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 405, 'name': 'sweat_pants'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 406, 'name': 'sweater'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 407, 'name': 'sweatshirt'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 408, 'name': 'sword'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 409, 'name': 'table'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 410, 'name': 'table_lamp'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 411, 'name': 'tablecloth'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 412, 'name': 'tag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 413, 'name': 'army_tank'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 414, 'name': 'tank_top_(clothing)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 415, 'name': 'tape_(sticky_cloth_or_paper)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 416, 'name': 'tape_measure'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 417, 'name': 'tarp'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 418, 'name': 'teacup'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 419, 'name': 'teakettle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 420, 'name': 'teapot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 421, 'name': 'telephone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 422, 'name': 'television_set'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 423, 'name': 'tennis_racket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 424, 'name': 'thermometer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 425, 'name': 'thermos_bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 426, 'name': 'thread'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 427, 'name': 'tiger'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 428, 'name': 'tinfoil'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 429, 'name': 'tissue_paper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 430, 'name': 'toast_(food)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 431, 'name': 'toaster'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 432, 'name': 'tongs'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 433, 'name': 'toolbox'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 434, 'name': 'toothbrush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 435, 'name': 'toothpaste'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 436, 'name': 'toothpick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 437, 'name': 'cover'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 438, 'name': 'tow_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 439, 'name': 'towel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 440, 'name': 'toy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 441, 'name': 'tractor_(farm_equipment)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 442, 'name': 'traffic_light'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 443, 'name': 'dirt_bike'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 444, 'name': 'trailer_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 445, 'name': 'train_(railroad_vehicle)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 446, 'name': 'tray'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 447, 'name': 'tripod'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 448, 'name': 'trousers'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 449, 'name': 'truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 450, 'name': 'turkey_(bird)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 451, 'name': 'turtle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 452, 'name': 'umbrella'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 453, 'name': 'underwear'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 454, 'name': 'vacuum_cleaner'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 455, 'name': 'violin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 456, 'name': 'volleyball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 457, 'name': 'waffle_iron'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 458, 'name': 'walking_stick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 459, 'name': 'wallet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 460, 'name': 'walrus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 461, 'name': 'wardrobe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 462, 'name': 'automatic_washer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 463, 'name': 'watch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 464, 'name': 'water_bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 465, 'name': 'water_faucet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 466, 'name': 'water_filter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 467, 'name': 'water_jug'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 468, 'name': 'water_scooter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 469, 'name': 'water_ski'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 470, 'name': 'watermelon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 471, 'name': 'wheel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 472, 'name': 'wheelchair'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 473, 'name': 'wig'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 474, 'name': 'wind_chime'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 475, 'name': 'windshield_wiper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 476, 'name': 'wine_bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 477, 'name': 'wineglass'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 478, 'name': 'wooden_spoon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 479, 'name': 'wrench'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 480, 'name': 'yacht'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 481, 'name': 'zebra'} ,]  





BURST_CATEGORIES = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 2, 'name': 'aerosol_can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 4, 'name': 'airplane'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 16, 'name': 'apricot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 17, 'name': 'apron'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 20, 'name': 'armchair'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 24, 'name': 'trash_can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 25, 'name': 'ashtray'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 27, 'name': 'atomizer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 31, 'name': 'ax'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 32, 'name': 'baby_buggy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 34, 'name': 'backpack'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 35, 'name': 'handbag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 36, 'name': 'suitcase'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 37, 'name': 'bagel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 38, 'name': 'bagpipe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 41, 'name': 'ball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 43, 'name': 'balloon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 45, 'name': 'banana'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 47, 'name': 'bandage'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 50, 'name': 'banner'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 51, 'name': 'barbell'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 55, 'name': 'barrow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 58, 'name': 'baseball_bat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 60, 'name': 'baseball_glove'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 61, 'name': 'basket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 62, 'name': 'basketball_hoop'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 63, 'name': 'basketball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 66, 'name': 'bath_mat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 74, 'name': 'beaker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 77, 'name': 'beanie'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 78, 'name': 'bear'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 79, 'name': 'bed'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 80, 'name': 'bedspread'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 81, 'name': 'cow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 83, 'name': 'beeper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 85, 'name': 'beer_can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 87, 'name': 'bell'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 88, 'name': 'bell_pepper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 91, 'name': 'bench'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 93, 'name': 'bib'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 95, 'name': 'bicycle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 97, 'name': 'binder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 98, 'name': 'binoculars'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 99, 'name': 'bird'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 100, 'name': 'birdfeeder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 102, 'name': 'birdcage'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 103, 'name': 'birdhouse'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 106, 'name': 'biscuit_(bread)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 108, 'name': 'black_sheep'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 110, 'name': 'blanket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 112, 'name': 'blender'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 117, 'name': 'gameboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 118, 'name': 'boat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 126, 'name': 'book'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 127, 'name': 'book_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 129, 'name': 'booklet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 133, 'name': 'bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 134, 'name': 'bottle_opener'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 135, 'name': 'bouquet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 136, 'name': 'bow_(weapon)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 139, 'name': 'bowl'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 146, 'name': 'bracelet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 152, 'name': 'briefcase'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 156, 'name': 'broom'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 160, 'name': 'bucket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 162, 'name': 'bull'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 170, 'name': 'bun'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 172, 'name': 'buoy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 174, 'name': 'bus_(vehicle)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 175, 'name': 'business_card'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 176, 'name': 'butcher_knife'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 180, 'name': 'cab_(taxi)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 183, 'name': 'cabinet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 187, 'name': 'calendar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 188, 'name': 'calf'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 189, 'name': 'camcorder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 190, 'name': 'camel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 191, 'name': 'camera'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 194, 'name': 'can'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 197, 'name': 'candle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 202, 'name': 'canister'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 204, 'name': 'canoe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 206, 'name': 'canteen'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 208, 'name': 'bottle_cap'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 209, 'name': 'cape'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 211, 'name': 'car_(automobile)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 212, 'name': 'railcar_(part_of_a_train)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 214, 'name': 'car_battery'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 216, 'name': 'card'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 217, 'name': 'cardigan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 221, 'name': 'carrot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 222, 'name': 'tote_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 223, 'name': 'cart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 224, 'name': 'carton'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 229, 'name': 'cat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 235, 'name': 'cellular_telephone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 236, 'name': 'chain_mail'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 237, 'name': 'chair'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 247, 'name': 'chicken_(animal)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 254, 'name': 'crisp_(potato_chip)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 256, 'name': 'chocolate_bar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 261, 'name': 'chopping_board'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 262, 'name': 'chopstick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 266, 'name': 'cigar_box'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 267, 'name': 'cigarette'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 268, 'name': 'cigarette_case'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 274, 'name': 'clip'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 275, 'name': 'clipboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 276, 'name': 'clock'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 278, 'name': 'clothes_hamper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 280, 'name': 'clutch_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 282, 'name': 'coat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 283, 'name': 'coat_hanger'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 285, 'name': 'cock'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 287, 'name': 'coffee_filter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 289, 'name': 'coffee_table'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 290, 'name': 'coffeepot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 292, 'name': 'coin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 297, 'name': 'pacifier'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 299, 'name': 'computer_keyboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 301, 'name': 'cone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 302, 'name': 'control'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 303, 'name': 'convertible_(automobile)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 307, 'name': 'cooking_utensil'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 308, 'name': 'cooler_(for_food)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 312, 'name': 'edible_corn'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 314, 'name': 'cornet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 323, 'name': 'cowboy_hat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 324, 'name': 'crab_(animal)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 325, 'name': 'cracker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 327, 'name': 'crate'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 336, 'name': 'crow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 341, 'name': 'crumb'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 342, 'name': 'crutch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 343, 'name': 'cub_(animal)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 344, 'name': 'cube'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 345, 'name': 'cucumber'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 347, 'name': 'cup'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 349, 'name': 'cupcake'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 352, 'name': 'curtain'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 353, 'name': 'cushion'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 355, 'name': 'cutting_tool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 356, 'name': 'cylinder'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 357, 'name': 'cymbal'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 363, 'name': 'deer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 365, 'name': 'desk'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 369, 'name': 'die'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 371, 'name': 'dining_table'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 373, 'name': 'dish'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 380, 'name': 'dispenser'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 382, 'name': 'dog'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 391, 'name': 'doormat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 395, 'name': 'drawer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 398, 'name': 'dress_hat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 403, 'name': 'drone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 405, 'name': 'drum_(musical_instrument)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 406, 'name': 'drumstick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 407, 'name': 'duck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 408, 'name': 'duckling'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 410, 'name': 'duffel_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 413, 'name': 'dustpan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 415, 'name': 'eagle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 416, 'name': 'earphone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 418, 'name': 'earring'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 422, 'name': 'egg'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 425, 'name': 'eggbeater'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 428, 'name': 'refrigerator'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 429, 'name': 'elephant'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 431, 'name': 'envelope'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 436, 'name': 'fan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 437, 'name': 'faucet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 440, 'name': 'Ferris_wheel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 446, 'name': 'file_(tool)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 448, 'name': 'fire_engine'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 453, 'name': 'fish'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 457, 'name': 'fishing_rod'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 458, 'name': 'flag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 463, 'name': 'flashlight'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 468, 'name': 'flute_glass'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 473, 'name': 'football_helmet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 474, 'name': 'footstool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 475, 'name': 'fork'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 480, 'name': 'frisbee'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 481, 'name': 'frog'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 484, 'name': 'frying_pan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 489, 'name': 'garbage'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 490, 'name': 'garbage_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 491, 'name': 'garden_hose'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 495, 'name': 'gasmask'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 499, 'name': 'giant_panda'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 500, 'name': 'gift_wrap'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 502, 'name': 'giraffe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 506, 'name': 'glove'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 507, 'name': 'goat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 508, 'name': 'goggles'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 510, 'name': 'golf_club'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 511, 'name': 'golfcart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 513, 'name': 'goose'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 514, 'name': 'gorilla'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 529, 'name': 'grocery_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 531, 'name': 'guitar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 533, 'name': 'gun'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 534, 'name': 'hair_spray'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 535, 'name': 'hairbrush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 539, 'name': 'hamburger'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 540, 'name': 'hammer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 543, 'name': 'hamster'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 544, 'name': 'hair_dryer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 546, 'name': 'hand_towel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 547, 'name': 'handcart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 548, 'name': 'handcuff'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 549, 'name': 'handkerchief'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 550, 'name': 'handle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 551, 'name': 'handsaw'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 553, 'name': 'harmonium'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 554, 'name': 'hat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 561, 'name': 'headscarf'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 567, 'name': 'helicopter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 568, 'name': 'helmet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 569, 'name': 'heron'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 572, 'name': 'hippopotamus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 573, 'name': 'hockey_stick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 574, 'name': 'hog'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 579, 'name': 'horse'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 580, 'name': 'hose'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 588, 'name': 'polar_bear'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 589, 'name': 'icecream'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 595, 'name': 'igniter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 598, 'name': 'iPod'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 599, 'name': 'iron_(for_clothing)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 601, 'name': 'jacket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 603, 'name': 'jean'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 604, 'name': 'jeep'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 606, 'name': 'jersey'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 611, 'name': 'kayak'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 614, 'name': 'kettle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 621, 'name': 'kite'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 625, 'name': 'knife'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 627, 'name': 'knitting_needle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 628, 'name': 'knob'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 633, 'name': 'ladle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 637, 'name': 'lamp'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 641, 'name': 'lanyard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 642, 'name': 'laptop_computer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 644, 'name': 'latch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 645, 'name': 'lawn_mower'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 649, 'name': 'lemon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 651, 'name': 'lettuce'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 653, 'name': 'life_buoy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 654, 'name': 'life_jacket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 660, 'name': 'lion'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 664, 'name': 'lizard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 666, 'name': 'log'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 672, 'name': 'magazine'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 675, 'name': 'mailbox_(at_home)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 676, 'name': 'mallet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 682, 'name': 'marker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 689, 'name': 'mat_(gym_equipment)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 690, 'name': 'matchbox'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 691, 'name': 'mattress'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 692, 'name': 'measuring_cup'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 693, 'name': 'measuring_stick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 695, 'name': 'medicine'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 697, 'name': 'microphone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 702, 'name': 'minivan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 704, 'name': 'mirror'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 706, 'name': 'mixer_(kitchen_tool)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 707, 'name': 'money'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 708, 'name': 'monitor_(computer_equipment) computer_monitor'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 709, 'name': 'monkey'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 711, 'name': 'motor_scooter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 713, 'name': 'motorboat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 714, 'name': 'motorcycle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 716, 'name': 'mouse_(animal_rodent)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 717, 'name': 'mouse_(computer_equipment)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 723, 'name': 'musical_instrument'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 726, 'name': 'napkin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 728, 'name': 'necklace'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 731, 'name': 'nest'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 732, 'name': 'newsstand'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 736, 'name': 'notebook'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 737, 'name': 'notepad'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 738, 'name': 'nut'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 740, 'name': 'oar'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 746, 'name': 'onion'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 747, 'name': 'orange_(fruit)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 754, 'name': 'packet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 757, 'name': 'paddle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 759, 'name': 'paintbox'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 760, 'name': 'paintbrush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 761, 'name': 'painting'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 762, 'name': 'pajamas'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 763, 'name': 'palette'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 764, 'name': 'pan_(for_cooking)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 771, 'name': 'paper_towel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 774, 'name': 'parachute'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 780, 'name': 'parrot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 790, 'name': 'peeler_(tool_for_fruit_and_vegetables)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 792, 'name': 'pelican'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 793, 'name': 'pen'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 794, 'name': 'pencil'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 798, 'name': 'penguin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 802, 'name': 'pepper_mill'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 804, 'name': 'persimmon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 805, 'name': 'person'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 807, 'name': 'petfood'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 810, 'name': 'phonograph_record'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 811, 'name': 'piano'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 812, 'name': 'pickle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 813, 'name': 'pickup_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 815, 'name': 'pigeon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 817, 'name': 'pillow'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 821, 'name': 'ping-pong_ball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 823, 'name': 'tobacco_pipe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 824, 'name': 'pipe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 825, 'name': 'pistol'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 827, 'name': 'pitcher_(vessel_for_liquid)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 829, 'name': 'pizza'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 831, 'name': 'plate'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 835, 'name': 'pliers'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 840, 'name': 'pole'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 844, 'name': 'pony'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 850, 'name': 'poster'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 851, 'name': 'pot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 852, 'name': 'flowerpot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 853, 'name': 'potato'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 856, 'name': 'pouch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 857, 'name': 'power_shovel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 868, 'name': 'pumpkin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 870, 'name': 'puppet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 875, 'name': 'rabbit'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 876, 'name': 'race_car'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 877, 'name': 'racket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 882, 'name': 'raft'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 883, 'name': 'rag_doll'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 887, 'name': 'rat'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 888, 'name': 'razorblade'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 893, 'name': 'record_player'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 896, 'name': 'remote_control'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 897, 'name': 'rhinoceros'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 899, 'name': 'rifle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 910, 'name': 'rubber_band'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 911, 'name': 'runner_(carpet)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 914, 'name': 'saddle_blanket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 924, 'name': 'saltshaker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 925, 'name': 'sandal_(type_of_shoe)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 926, 'name': 'sandwich'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 928, 'name': 'saucepan'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 932, 'name': 'saxophone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 935, 'name': 'scarf'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 936, 'name': 'school_bus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 937, 'name': 'scissors'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 938, 'name': 'scoreboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 940, 'name': 'scraper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 942, 'name': 'screwdriver'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 943, 'name': 'scrubbing_brush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 944, 'name': 'sculpture'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 950, 'name': 'serving_dish'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 952, 'name': 'shaker'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 953, 'name': 'shampoo'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 954, 'name': 'shark'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 955, 'name': 'sharpener'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 957, 'name': 'shaver_(electric)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 959, 'name': 'shawl'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 960, 'name': 'shears'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 961, 'name': 'sheep'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 964, 'name': 'shield'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 965, 'name': 'shirt'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 966, 'name': 'shoe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 968, 'name': 'shopping_cart'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 969, 'name': 'short_pants'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 971, 'name': 'shoulder_bag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 972, 'name': 'shovel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 976, 'name': 'sieve'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 980, 'name': 'skateboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 982, 'name': 'ski'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 985, 'name': 'ski_pole'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 986, 'name': 'skirt'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 987, 'name': 'sled'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 990, 'name': 'slipper_(footwear)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 996, 'name': 'soap'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 998, 'name': 'sock'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1001, 'name': 'sofa'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1007, 'name': 'soupspoon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1012, 'name': 'spatula'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1014, 'name': 'spectacles'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1016, 'name': 'spider'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1017, 'name': 'sponge'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1018, 'name': 'spoon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1020, 'name': 'spotlight'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1021, 'name': 'squirrel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1024, 'name': 'statue_(sculpture)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1028, 'name': 'steering_wheel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1031, 'name': 'step_stool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1037, 'name': 'stool'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1040, 'name': 'stove'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1043, 'name': 'straw_(for_drinking)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1044, 'name': 'strawberry'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1045, 'name': 'street_sign'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1048, 'name': 'stylus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1050, 'name': 'sugar_bowl'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1054, 'name': 'sunglasses'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1056, 'name': 'sunscreen'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1057, 'name': 'surfboard'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1059, 'name': 'mop'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1060, 'name': 'sweat_pants'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1062, 'name': 'sweater'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1063, 'name': 'sweatshirt'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1066, 'name': 'sword'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1070, 'name': 'table'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1071, 'name': 'table_lamp'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1072, 'name': 'tablecloth'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1075, 'name': 'tag'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1078, 'name': 'army_tank'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1080, 'name': 'tank_top_(clothing)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1081, 'name': 'tape_(sticky_cloth_or_paper)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1082, 'name': 'tape_measure'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1084, 'name': 'tarp'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1088, 'name': 'teacup'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1089, 'name': 'teakettle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1090, 'name': 'teapot'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1092, 'name': 'telephone'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1097, 'name': 'television_set'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1099, 'name': 'tennis_racket'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1101, 'name': 'thermometer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1102, 'name': 'thermos_bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1105, 'name': 'thread'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1108, 'name': 'tiger'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1111, 'name': 'tinfoil'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1113, 'name': 'tissue_paper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1114, 'name': 'toast_(food)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1115, 'name': 'toaster'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1120, 'name': 'tongs'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1121, 'name': 'toolbox'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1122, 'name': 'toothbrush'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1123, 'name': 'toothpaste'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1124, 'name': 'toothpick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1125, 'name': 'cover'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1127, 'name': 'tow_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1128, 'name': 'towel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1130, 'name': 'toy'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1131, 'name': 'tractor_(farm_equipment)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1132, 'name': 'traffic_light'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1133, 'name': 'dirt_bike'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1134, 'name': 'trailer_truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1135, 'name': 'train_(railroad_vehicle)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1137, 'name': 'tray'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1142, 'name': 'tripod'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1143, 'name': 'trousers'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1144, 'name': 'truck'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1149, 'name': 'turkey_(bird)'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1152, 'name': 'turtle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1155, 'name': 'umbrella'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1156, 'name': 'underwear'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1160, 'name': 'vacuum_cleaner'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1167, 'name': 'violin'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1169, 'name': 'volleyball'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1172, 'name': 'waffle_iron'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1175, 'name': 'walking_stick'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1178, 'name': 'wallet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1179, 'name': 'walrus'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1180, 'name': 'wardrobe'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1182, 'name': 'automatic_washer'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1183, 'name': 'watch'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1184, 'name': 'water_bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1186, 'name': 'water_faucet'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1187, 'name': 'water_filter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1189, 'name': 'water_jug'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1191, 'name': 'water_scooter'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1192, 'name': 'water_ski'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1195, 'name': 'watermelon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1201, 'name': 'wheel'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1202, 'name': 'wheelchair'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1207, 'name': 'wig'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1208, 'name': 'wind_chime'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1211, 'name': 'windshield_wiper'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1213, 'name': 'wine_bottle'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1215, 'name': 'wineglass'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1220, 'name': 'wooden_spoon'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1222, 'name': 'wrench'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1225, 'name': 'yacht'} ,
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1229, 'name': 'zebra'} ,
    ]  



def _get_burst_image_meta():

    thing_ids = [k["id"] for k in BURST_CATEGORIES]
    assert len(thing_ids) == 482, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BURST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_burst_video_meta():
    thing_ids = [k["id"] for k in BURST_VIDEO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in BURST_VIDEO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 482, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in BURST_VIDEO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, dataset_name_in_dict="burst_video"):
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
        record["exhaustive_category_ids"] = vid_dict["exhaustive_category_ids"]
        record["not_exhaustive_category_ids"] = vid_dict["not_exhaustive_category_ids"]
        record["neg_category_ids"] = vid_dict["neg_category_ids"]
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
                # BURST has no box annotations, need to generate from mask or use TAO annotations 
                # if not (_segm and _segm[frame_idx]):
                #     continue

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
        record['has_mask'] = True
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


def register_burst_video(name, metadata, json_file, image_root, dataset_name_in_dict="burst_video"):
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
