# Copyright (c) 2024 ByteDance. All Rights Reserved.
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
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]




LVVIS_CATEGORIES = [
        {"color": [220, 20, 60], "isthing": 1, 'id': 1, 'name': 'accordion', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 2, 'name': 'action_camera', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 3, 'name': 'aerosol_can', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 4, 'name': 'air_conditioner', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 5, 'name': 'air_fryer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 6, 'name': 'airplane', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 7, 'name': 'alarm_clock', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 8, 'name': 'alcohol', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 9, 'name': 'alcohol_lamp', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 10, 'name': 'alligator', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 11, 'name': 'almond', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 12, 'name': 'ambulance', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 13, 'name': 'amplifier', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 14, 'name': 'anklet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 15, 'name': 'antelope', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 16, 'name': 'antenna', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 17, 'name': 'apple', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 18, 'name': 'applesauce', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 19, 'name': 'apricot', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 20, 'name': 'apron', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 21, 'name': 'armband', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 22, 'name': 'armchair', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 23, 'name': 'armoire', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 24, 'name': 'armor', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 25, 'name': 'army_tank', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 26, 'name': 'artichoke', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 27, 'name': 'ashtray', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 28, 'name': 'asparagus', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 29, 'name': 'atomizer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 30, 'name': 'automatic_washer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 31, 'name': 'avocado', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 32, 'name': 'award', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 33, 'name': 'awning', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 34, 'name': 'ax', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 35, 'name': 'baboon', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 36, 'name': 'baby_buggy', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 37, 'name': 'backhoe', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 38, 'name': 'backpack', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 39, 'name': 'badger', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 40, 'name': 'bagpipe', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 41, 'name': 'baguet', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 42, 'name': 'bait', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 43, 'name': 'balance_scale', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 44, 'name': 'balances', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 45, 'name': 'ball', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 46, 'name': 'ballet_skirt', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 47, 'name': 'balloon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 48, 'name': 'ballpoint_pen', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 49, 'name': 'bamboo', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 50, 'name': 'bamboo_shoots', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 51, 'name': 'banana', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 52, 'name': 'band_aid', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 53, 'name': 'bandage', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 54, 'name': 'banjo', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 55, 'name': 'banner', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 56, 'name': 'barbell', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 57, 'name': 'barcode', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 58, 'name': 'barrel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 59, 'name': 'barrette', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 60, 'name': 'barrow', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 61, 'name': 'baseball', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 62, 'name': 'baseball_bat', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 63, 'name': 'baseball_cap', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 64, 'name': 'baseball_glove', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 65, 'name': 'basket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 66, 'name': 'basketball', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 67, 'name': 'basketball_backboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 68, 'name': 'bass_guitar', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 69, 'name': 'bass_horn', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 70, 'name': 'bassoon', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 71, 'name': 'bat_(animal)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 72, 'name': 'bath_mat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 73, 'name': 'bath_towel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 74, 'name': 'bathrobe', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 75, 'name': 'bathtub', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 76, 'name': 'battery', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 77, 'name': 'beach_towel', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 78, 'name': 'beachball', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 79, 'name': 'beaker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 80, 'name': 'bean_curd', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 81, 'name': 'beanbag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 82, 'name': 'beanie', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 83, 'name': 'bear', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 84, 'name': 'bed', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 85, 'name': 'bedpan', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 86, 'name': 'beef_(food)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 87, 'name': 'beeper', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 88, 'name': 'beer_bottle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 89, 'name': 'beer_can', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 90, 'name': 'beetle', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 91, 'name': 'bell', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 92, 'name': 'bell_pepper', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 93, 'name': 'belt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 94, 'name': 'belt_buckle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 95, 'name': 'beluga_whale', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 96, 'name': 'bench', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 97, 'name': 'beret', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 98, 'name': 'bib', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 99, 'name': 'bible', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 100, 'name': 'bicycle', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 101, 'name': 'bicycle_pump', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 102, 'name': 'billards', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 103, 'name': 'billboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 104, 'name': 'binder', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 105, 'name': 'binder_clip', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 106, 'name': 'binoculars', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 107, 'name': 'bird', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 108, 'name': 'birdbath', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 109, 'name': 'birdcage', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 110, 'name': 'birdfeeder', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 111, 'name': 'birdhouse', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 112, 'name': 'birthday_cake', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 113, 'name': 'birthday_card', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 114, 'name': 'blackboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 115, 'name': 'blanket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 116, 'name': 'blazer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 117, 'name': 'blender', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 118, 'name': 'blimp', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 119, 'name': 'blouse', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 120, 'name': 'blue_whale', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 121, 'name': 'blueberry', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 122, 'name': 'board_eraser', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 123, 'name': 'boat', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 124, 'name': 'bobby_pin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 125, 'name': 'body_thermometer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 126, 'name': 'boiled_egg', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 127, 'name': 'bolo_tie', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 128, 'name': 'bolt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 129, 'name': 'bongos', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 130, 'name': 'book', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 131, 'name': 'bookcase', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 132, 'name': 'bookend', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 133, 'name': 'bookmark', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 134, 'name': 'boot', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 135, 'name': 'bottle', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 136, 'name': 'bottle_cap', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 137, 'name': 'bottle_opener', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 138, 'name': 'bouquet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 139, 'name': 'bow_(decorative_ribbons)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 140, 'name': 'bow_(weapon)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 141, 'name': 'bow-tie', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 142, 'name': 'bowl', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 143, 'name': 'bowler_hat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 144, 'name': 'bowling_ball', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 145, 'name': 'box', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 146, 'name': 'boxing_glove', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 147, 'name': 'bracelet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 148, 'name': 'brass_plaque', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 149, 'name': 'brassiere', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 150, 'name': 'bread', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 151, 'name': 'bread-bin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 152, 'name': 'briefcase', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 153, 'name': 'broccoli', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 154, 'name': 'broom', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 155, 'name': 'bubble_gum', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 156, 'name': 'bucket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 157, 'name': 'building_blocks', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 158, 'name': 'bulldog', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 159, 'name': 'bulldozer', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 160, 'name': 'bulletin_board', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 161, 'name': 'bulletproof_vest', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 162, 'name': 'bullhorn', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 163, 'name': 'bun', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 164, 'name': 'bunk_bed', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 165, 'name': 'buoy', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 166, 'name': 'burette', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 167, 'name': 'bus_(vehicle)', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 168, 'name': 'butter', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 169, 'name': 'butterfly', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 170, 'name': 'button', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 171, 'name': 'cab_(taxi)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 172, 'name': 'cabbage', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 173, 'name': 'cabinet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 174, 'name': 'cable_car', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 175, 'name': 'cage', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 176, 'name': 'cake', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 177, 'name': 'calculator', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 178, 'name': 'calendar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 179, 'name': 'camcorder', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 180, 'name': 'camel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 181, 'name': 'camera', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 182, 'name': 'camera_lens', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 183, 'name': 'camera_tripod', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 184, 'name': 'camper_(vehicle)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 185, 'name': 'can', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 186, 'name': 'can_opener', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 187, 'name': 'candle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 188, 'name': 'candle_holder', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 189, 'name': 'candy_bar', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 190, 'name': 'candy_cane', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 191, 'name': 'canister', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 192, 'name': 'canoe', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 193, 'name': 'cantaloup', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 194, 'name': 'canteen', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 195, 'name': 'canvas_bag', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 196, 'name': 'car_(automobile)', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 197, 'name': 'car_battery', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 198, 'name': 'car_jack', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 199, 'name': 'car_odometer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 200, 'name': 'carabiner', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 201, 'name': 'card', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 202, 'name': 'card_game', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 203, 'name': 'cardigan', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 204, 'name': 'cargo_ship', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 205, 'name': 'carnation', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 206, 'name': 'carpet', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 207, 'name': 'carrot', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 208, 'name': 'cart', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 209, 'name': 'carton', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 210, 'name': 'casserole', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 211, 'name': 'cassette', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 212, 'name': 'cassette_player', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 213, 'name': 'cat', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 214, 'name': 'cauliflower', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 215, 'name': 'CD', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 216, 'name': 'cd_player', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 217, 'name': 'cell_phone_charger', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 218, 'name': 'cello', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 219, 'name': 'cellular_telephone', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 220, 'name': 'centipede', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 221, 'name': 'centrifuge', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 222, 'name': 'certificate', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 223, 'name': 'chain_mail', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 224, 'name': 'chair', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 225, 'name': 'chaise_longue', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 226, 'name': 'chalice', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 227, 'name': 'chalk', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 228, 'name': 'chandelier', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 229, 'name': 'chap', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 230, 'name': 'charger', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 231, 'name': 'checkbook', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 232, 'name': 'cheese_curls', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 233, 'name': 'cheetah', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 234, 'name': 'chef_hat', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 235, 'name': 'cheongsam', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 236, 'name': 'cherry', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 237, 'name': 'chessboard', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 238, 'name': 'chestnut', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 239, 'name': 'chewing_gum', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 240, 'name': 'chicken_(animal)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 241, 'name': 'chickpea', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 242, 'name': 'chili_(vegetable)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 243, 'name': 'chime', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 244, 'name': 'chinaware', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 245, 'name': 'chisel', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 246, 'name': 'chocolate_bar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 247, 'name': 'choker', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 248, 'name': 'chopping_board', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 249, 'name': 'chopstick', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 250, 'name': 'christmas_tree', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 251, 'name': 'cicada', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 252, 'name': 'cigar_box', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 253, 'name': 'cigarette', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 254, 'name': 'clam', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 255, 'name': 'clarinet', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 256, 'name': 'clasp', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 257, 'name': 'claw_hammer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 258, 'name': 'cleat_(for_securing_rope)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 259, 'name': 'clippers_(for_plants)', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 260, 'name': 'cloak', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 261, 'name': 'clock', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 262, 'name': 'clock_tower', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 263, 'name': 'clothes_hamper', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 264, 'name': 'clothespin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 265, 'name': 'coaster', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 266, 'name': 'coat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 267, 'name': 'coat_hanger', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 268, 'name': 'coatrack', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 269, 'name': 'cockatoo', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 270, 'name': 'cockroach', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 271, 'name': 'coconut', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 272, 'name': 'cod', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 273, 'name': 'coffee_maker', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 274, 'name': 'coffee_mug', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 275, 'name': 'coffeepot', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 276, 'name': 'coin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 277, 'name': 'combination_lock', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 278, 'name': 'comic_book', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 279, 'name': 'compass', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 280, 'name': 'compass_(drawing_tool)', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 281, 'name': 'computer_box', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 282, 'name': 'computer_keyboard', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 283, 'name': 'condiment', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 284, 'name': 'cone', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 285, 'name': 'convertible_(automobile)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 286, 'name': 'cooler_(for_food)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 287, 'name': 'cork_(bottle_plug)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 288, 'name': 'corkscrew', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 289, 'name': 'cornet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 290, 'name': 'correction_fluid', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 291, 'name': 'correction_tape', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 292, 'name': 'cotton_swab', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 293, 'name': 'cougar', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 294, 'name': 'cover', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 295, 'name': 'coverall', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 296, 'name': 'cow', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 297, 'name': 'cowboy_hat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 298, 'name': 'crab_(animal)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 299, 'name': 'cranberry', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 300, 'name': 'crane', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 301, 'name': 'crate', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 302, 'name': 'crawfish', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 303, 'name': 'crayon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 304, 'name': 'cream_pitcher', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 305, 'name': 'crib', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 306, 'name': 'crisp_(potato_chip)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 307, 'name': 'crock_pot', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 308, 'name': 'crossbar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 309, 'name': 'crowbar', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 310, 'name': 'crucifix', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 311, 'name': 'cruise_ship', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 312, 'name': 'crutch', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 313, 'name': 'cucumber', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 314, 'name': 'cup', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 315, 'name': 'cupcake', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 316, 'name': 'curling_iron', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 317, 'name': 'curtain', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 318, 'name': 'cushion', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 319, 'name': 'cuttlefish', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 320, 'name': 'cymbal', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 321, 'name': 'dagger', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 322, 'name': 'dalmatian', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 323, 'name': 'dartboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 324, 'name': 'darts', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 325, 'name': 'date_(fruit)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 326, 'name': 'deadbolt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 327, 'name': 'deck_chair', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 328, 'name': 'deer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 329, 'name': 'defibrillator', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 330, 'name': 'dehumidifier', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 331, 'name': 'desk', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 332, 'name': 'desk_chair', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 333, 'name': 'detergent', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 334, 'name': 'diaper', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 335, 'name': 'die', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 336, 'name': 'dining_table', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 337, 'name': 'dirt_bike', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 338, 'name': 'discus', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 339, 'name': 'dish', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 340, 'name': 'dish_antenna', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 341, 'name': 'dishrag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 342, 'name': 'dishtowel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 343, 'name': 'dishwasher', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 344, 'name': 'dispenser', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 345, 'name': 'diving_board', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 346, 'name': 'dog', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 347, 'name': 'dog_collar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 348, 'name': 'doll', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 349, 'name': 'dollar', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 350, 'name': 'dolphin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 351, 'name': 'domestic_ass', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 352, 'name': 'doorknob', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 353, 'name': 'doormat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 354, 'name': 'double-sided_tape', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 355, 'name': 'doughnut', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 356, 'name': 'dove', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 357, 'name': 'dragonfly', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 358, 'name': 'dragonfruit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 359, 'name': 'drawer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 360, 'name': 'drawing_board', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 361, 'name': 'dress', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 362, 'name': 'dress_hat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 363, 'name': 'dresser', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 364, 'name': 'drill', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 365, 'name': 'drill_bit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 366, 'name': 'drone', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 367, 'name': 'dropper', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 368, 'name': 'drum_(musical_instrument)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 369, 'name': 'drum_set', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 370, 'name': 'drumstick', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 371, 'name': 'duck', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 372, 'name': 'duct_tape', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 373, 'name': 'duffel_bag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 374, 'name': 'dulcimer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 375, 'name': 'dumbbell', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 376, 'name': 'dumpling', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 377, 'name': 'dumpster', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 378, 'name': 'durian', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 379, 'name': 'dustpan', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 380, 'name': 'e-cigarette', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 381, 'name': 'e-reader', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 382, 'name': 'eagle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 383, 'name': 'earmuffs', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 384, 'name': 'earphone', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 385, 'name': 'earplug', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 386, 'name': 'earring', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 387, 'name': 'easel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 388, 'name': 'eclair', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 389, 'name': 'eel', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 390, 'name': 'egg', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 391, 'name': 'egg_roll', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 392, 'name': 'egg_tart', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 393, 'name': 'egg_yolk', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 394, 'name': 'eggbeater', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 395, 'name': 'eggplant', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 396, 'name': 'electric_bicycle', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 397, 'name': 'electric_chair', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 398, 'name': 'electric_drill', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 399, 'name': 'electric_heater', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 400, 'name': 'electric_kettle', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 401, 'name': 'elephant', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 402, 'name': 'envelope', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 403, 'name': 'eraser', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 404, 'name': 'escargot', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 405, 'name': 'excavator', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 406, 'name': 'external_hard_drive', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 407, 'name': 'eye_liner', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 408, 'name': 'eye_shadow', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 409, 'name': 'eyepatch', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 410, 'name': 'face_mask', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 411, 'name': 'falcon', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 412, 'name': 'fan', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 413, 'name': 'faucet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 414, 'name': 'fedora', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 415, 'name': 'ferret', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 416, 'name': 'fig_(fruit)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 417, 'name': 'file_(tool)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 418, 'name': 'file_folder', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 419, 'name': 'fire_alarm', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 420, 'name': 'fire_engine', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 421, 'name': 'fire_extinguisher', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 422, 'name': 'fire_truck', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 423, 'name': 'firefly', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 424, 'name': 'fireplace', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 425, 'name': 'fireplug', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 426, 'name': 'first-aid_kit', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 427, 'name': 'fish', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 428, 'name': 'fishbowl', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 429, 'name': 'fishing_rod', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 430, 'name': 'flag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 431, 'name': 'flagpole', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 432, 'name': 'flamingo', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 433, 'name': 'flash_drive', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 434, 'name': 'flashlight', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 435, 'name': 'flea', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 436, 'name': 'flip-flop_(sandal)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 437, 'name': 'flower_arrangement', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 438, 'name': 'flowerpot', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 439, 'name': 'flute', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 440, 'name': 'fly', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 441, 'name': 'folding_chair', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 442, 'name': 'folding_knife', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 443, 'name': 'football_(american)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 444, 'name': 'football_helmet', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 445, 'name': 'footstool', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 446, 'name': 'fork', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 447, 'name': 'forklift', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 448, 'name': 'fox', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 449, 'name': 'fragrance', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 450, 'name': 'freight_car', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 451, 'name': 'french_horn', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 452, 'name': 'fridge_magnet', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 453, 'name': 'fried_chicken', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 454, 'name': 'frisbee', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 455, 'name': 'frog', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 456, 'name': 'fruit_juice', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 457, 'name': 'frying_pan', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 458, 'name': 'fume_hood', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 459, 'name': 'funnel', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 460, 'name': 'game_console', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 461, 'name': 'gameboard', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 462, 'name': 'gamepad', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 463, 'name': 'gaming_chairs', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 464, 'name': 'garbage_truck', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 465, 'name': 'garlic', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 466, 'name': 'gas_pipe', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 467, 'name': 'gas_stove', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 468, 'name': 'gasmask', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 469, 'name': 'gazelle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 470, 'name': 'gemstone', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 471, 'name': 'generator', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 472, 'name': 'giant_panda', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 473, 'name': 'gift_wrap', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 474, 'name': 'giraffe', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 475, 'name': 'glass_(drink_container)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 476, 'name': 'glider', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 477, 'name': 'globe', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 478, 'name': 'glockenspiel', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 479, 'name': 'glove', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 480, 'name': 'glue', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 481, 'name': 'go-kart', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 482, 'name': 'goat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 483, 'name': 'goggles', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 484, 'name': 'goldfish', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 485, 'name': 'golf_ball', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 486, 'name': 'golf_club', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 487, 'name': 'golfcart', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 488, 'name': 'goose', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 489, 'name': 'gorilla', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 490, 'name': 'gourd', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 491, 'name': 'grape', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 492, 'name': 'grapefruit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 493, 'name': 'grasshopper', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 494, 'name': 'grater', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 495, 'name': 'gravestone', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 496, 'name': 'green_onion', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 497, 'name': 'green_plants_(potted_plants)', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 498, 'name': 'griddle', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 499, 'name': 'grill', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 500, 'name': 'guava', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 501, 'name': 'guitar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 502, 'name': 'gun', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 503, 'name': 'hair_curler', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 504, 'name': 'hair_dryer', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 505, 'name': 'hairbrush', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 506, 'name': 'hairnet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 507, 'name': 'hairpin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 508, 'name': 'halter_top', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 509, 'name': 'ham', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 510, 'name': 'hamburger', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 511, 'name': 'hammer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 512, 'name': 'hammock', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 513, 'name': 'hamper', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 514, 'name': 'hamster', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 515, 'name': 'hand_grips_strengthener', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 516, 'name': 'hand_towel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 517, 'name': 'handbag', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 518, 'name': 'handcart', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 519, 'name': 'handcuff', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 520, 'name': 'handkerchief', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 521, 'name': 'handle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 522, 'name': 'handsaw', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 523, 'name': 'hang_glider', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 524, 'name': 'hard_drive', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 525, 'name': 'hardback_book', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 526, 'name': 'harmonium', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 527, 'name': 'hat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 528, 'name': 'hatbox', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 529, 'name': 'headband', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 530, 'name': 'headboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 531, 'name': 'headlamp', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 532, 'name': 'headlight', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 533, 'name': 'headscarf', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 534, 'name': 'headset', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 535, 'name': 'heart', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 536, 'name': 'hedgehog', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 537, 'name': 'helicopter', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 538, 'name': 'helmet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 539, 'name': 'heron', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 540, 'name': 'high_heels', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 541, 'name': 'high_jump_standards', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 542, 'name': 'highchair', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 543, 'name': 'hinge', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 544, 'name': 'hippopotamus', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 545, 'name': 'hockey_stick', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 546, 'name': 'hog', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 547, 'name': 'honey', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 548, 'name': 'hoodie', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 549, 'name': 'hookah', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 550, 'name': 'horse', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 551, 'name': 'horse_buggy', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 552, 'name': 'hose', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 553, 'name': 'hot_dog', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 554, 'name': 'hot-air_balloon', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 555, 'name': 'hotplate', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 556, 'name': 'hourglass', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 557, 'name': 'houseboat', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 558, 'name': 'hovercraft', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 559, 'name': 'humidifier', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 560, 'name': 'hummingbird', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 561, 'name': 'hurdle', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 562, 'name': 'ice_pack', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 563, 'name': 'ice_skate', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 564, 'name': 'icecream', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 565, 'name': 'igniter', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 566, 'name': 'incense_burner', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 567, 'name': 'inflatable_bed', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 568, 'name': 'infusion_pump', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 569, 'name': 'iron_(for_clothing)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 570, 'name': 'ironing_board', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 571, 'name': 'jackal', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 572, 'name': 'jacket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 573, 'name': 'jackfruit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 574, 'name': 'jaguar', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 575, 'name': 'jar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 576, 'name': 'javelin', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 577, 'name': 'jean', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 578, 'name': 'jeep', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 579, 'name': 'jellyfish', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 580, 'name': 'jersey', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 581, 'name': 'jet_plane', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 582, 'name': 'joystick', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 583, 'name': 'jump_rope', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 584, 'name': 'kangaroo', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 585, 'name': 'kayak', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 586, 'name': 'keg', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 587, 'name': 'kennel', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 588, 'name': 'kettle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 589, 'name': 'kettlebell', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 590, 'name': 'key', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 591, 'name': 'keycard', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 592, 'name': 'kimono', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 593, 'name': 'kitchen_paper', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 594, 'name': 'kitchen_sink', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 595, 'name': 'kite', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 596, 'name': 'kiwi_fruit', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 597, 'name': 'knee_pad', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 598, 'name': 'knife', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 599, 'name': 'knob', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 600, 'name': 'koala', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 601, 'name': 'lab_coat', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 602, 'name': 'ladder', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 603, 'name': 'ladybug', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 604, 'name': 'lamb-chop', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 605, 'name': 'lamp', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 606, 'name': 'lamppost', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 607, 'name': 'lampshade', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 608, 'name': 'lantern', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 609, 'name': 'lanyard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 610, 'name': 'laptop_computer', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 611, 'name': 'lasagna', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 612, 'name': 'latch', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 613, 'name': 'lawn_mower', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 614, 'name': 'leather_shoes', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 615, 'name': 'legging_(clothing)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 616, 'name': 'lego', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 617, 'name': 'lemon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 618, 'name': 'leopard_cat', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 619, 'name': 'lettuce', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 620, 'name': 'level_(tools)', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 621, 'name': 'license_plate', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 622, 'name': 'life_buoy', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 623, 'name': 'life_jacket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 624, 'name': 'lightbulb', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 625, 'name': 'lighter', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 626, 'name': 'lighthouse', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 627, 'name': 'lightning_rod', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 628, 'name': 'lion', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 629, 'name': 'lip_balm', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 630, 'name': 'lipstick', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 631, 'name': 'liquor', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 632, 'name': 'lizard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 633, 'name': 'loader', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 634, 'name': 'lobster', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 635, 'name': 'locker', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 636, 'name': 'log', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 637, 'name': 'lollipop', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 638, 'name': 'long_jump_pit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 639, 'name': 'lychee', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 640, 'name': 'machine_gun', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 641, 'name': 'magazine', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 642, 'name': 'magic_cube', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 643, 'name': 'maglev', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 644, 'name': 'magnet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 645, 'name': 'magpie', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 646, 'name': 'mailbox_(at_home)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 647, 'name': 'mallard', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 648, 'name': 'mammoth', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 649, 'name': 'manatee', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 650, 'name': 'mandolin', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 651, 'name': 'mango', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 652, 'name': 'manhole', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 653, 'name': 'manual_(instruction_book)', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 654, 'name': 'map', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 655, 'name': 'marble', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 656, 'name': 'marker', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 657, 'name': 'marten', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 658, 'name': 'martini', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 659, 'name': 'mashed_potato', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 660, 'name': 'masher', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 661, 'name': 'mask', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 662, 'name': 'massage_chair', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 663, 'name': 'mast', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 664, 'name': 'mat_(gym_equipment)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 665, 'name': 'match', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 666, 'name': 'matchbox', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 667, 'name': 'mealworms', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 668, 'name': 'measuring_stick', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 669, 'name': 'meatball', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 670, 'name': 'mechanical_pencil', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 671, 'name': 'medal', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 672, 'name': 'megaphone', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 673, 'name': 'melon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 674, 'name': 'memo_pad', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 675, 'name': 'microphone', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 676, 'name': 'microscope', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 677, 'name': 'microwave_oven', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 678, 'name': 'milestone', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 679, 'name': 'milk', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 680, 'name': 'milk_can', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 681, 'name': 'milkshake', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 682, 'name': 'minivan', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 683, 'name': 'mint_candy', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 684, 'name': 'mirror', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 685, 'name': 'mitten', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 686, 'name': 'mixer_(kitchen_tool)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 687, 'name': 'mole', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 688, 'name': 'monitor_(computer_equipment)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 689, 'name': 'monkey', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 690, 'name': 'moose', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 691, 'name': 'mop', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 692, 'name': 'mosquito', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 693, 'name': 'motor_vehicle', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 694, 'name': 'motorcycle', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 695, 'name': 'mouse_(computer_equipment)', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 696, 'name': 'mousepad', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 697, 'name': 'mug', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 698, 'name': 'mushroom', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 699, 'name': 'music_box', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 700, 'name': 'music_stand', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 701, 'name': 'music_stool', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 702, 'name': 'nail_polish', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 703, 'name': 'nailfile', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 704, 'name': 'napkin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 705, 'name': 'nebulizer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 706, 'name': 'necklace', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 707, 'name': 'necktie', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 708, 'name': 'nectarine', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 709, 'name': 'needle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 710, 'name': 'nest', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 711, 'name': 'nightingale', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 712, 'name': 'nightshirt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 713, 'name': 'nightstand', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 714, 'name': 'noodle', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 715, 'name': 'nosebag_(for_animals)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 716, 'name': 'noseband_(for_animals)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 717, 'name': 'notebook', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 718, 'name': 'notepad', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 719, 'name': 'nut', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 720, 'name': 'nutcracker', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 721, 'name': 'oar', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 722, 'name': 'oboe', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 723, 'name': 'octopus_(animal)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 724, 'name': 'octopus_(food)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 725, 'name': 'oil_lamp', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 726, 'name': 'oil_tanker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 727, 'name': 'okra', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 728, 'name': 'onion', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 729, 'name': 'orange_(fruit)', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 730, 'name': 'organ', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 731, 'name': 'ostrich', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 732, 'name': 'otoscope', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 733, 'name': 'otter', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 734, 'name': 'oven', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 735, 'name': 'owl', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 736, 'name': 'oxygen_concentrator', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 737, 'name': 'oyster', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 738, 'name': 'pad', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 739, 'name': 'pad_(electronic_product)', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 740, 'name': 'paddle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 741, 'name': 'padlock', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 742, 'name': 'paint_brush', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 743, 'name': 'paintbrush', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 744, 'name': 'painting', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 745, 'name': 'palette', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 746, 'name': 'pan_(for_cooking)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 747, 'name': 'papaya', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 748, 'name': 'paper_bag', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 749, 'name': 'paper_clip', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 750, 'name': 'parachute', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 751, 'name': 'parasail_(sports)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 752, 'name': 'parchment', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 753, 'name': 'parka', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 754, 'name': 'parrot', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 755, 'name': 'passenger_ship', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 756, 'name': 'passion_fruit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 757, 'name': 'pasta_strainer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 758, 'name': 'pastry', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 759, 'name': 'peach', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 760, 'name': 'peacock', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 761, 'name': 'pear', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 762, 'name': 'peeler_(tool_for_fruit_and_vegetables)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 763, 'name': 'pegboard', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 764, 'name': 'pelican', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 765, 'name': 'pen', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 766, 'name': 'pencil', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 767, 'name': 'pencil_box', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 768, 'name': 'pencil_sharpener', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 769, 'name': 'pendulum', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 770, 'name': 'penguin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 771, 'name': 'pennant', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 772, 'name': 'persimmon', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 773, 'name': 'person', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 774, 'name': 'petri_dish', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 775, 'name': 'phonograph_record', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 776, 'name': 'piano', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 777, 'name': 'pickaxe', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 778, 'name': 'pickup_truck', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 779, 'name': 'picnic_basket', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 780, 'name': 'picture', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 781, 'name': 'pigeon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 782, 'name': 'piggy_bank', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 783, 'name': 'pill', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 784, 'name': 'pillow', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 785, 'name': 'pin_(non_jewelry)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 786, 'name': 'pineapple', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 787, 'name': 'pinecone', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 788, 'name': 'ping-pong_ball', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 789, 'name': 'pinwheel', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 790, 'name': 'pipe', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 791, 'name': 'pirate_flag', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 792, 'name': 'pistol', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 793, 'name': 'pizza', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 794, 'name': 'place_mat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 795, 'name': 'plastic_bag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 796, 'name': 'plate', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 797, 'name': 'platypus', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 798, 'name': 'playpen', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 799, 'name': 'pliers', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 800, 'name': 'plow_(farm_equipment)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 801, 'name': 'plume', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 802, 'name': 'pocket_watch', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 803, 'name': 'poker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 804, 'name': 'poker_chip', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 805, 'name': 'polar_bear', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 806, 'name': 'pole', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 807, 'name': 'police_car', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 808, 'name': 'polo_shirt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 809, 'name': 'pomegranate', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 810, 'name': 'pool_cue', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 811, 'name': 'pool_table', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 812, 'name': 'popcorn', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 813, 'name': 'popsicle', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 814, 'name': 'postcard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 815, 'name': 'poster', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 816, 'name': 'pot', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 817, 'name': 'potato', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 818, 'name': 'potholder', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 819, 'name': 'pouch', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 820, 'name': 'power_bank', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 821, 'name': 'power_drill', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 822, 'name': 'power_saw', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 823, 'name': 'praying_mantis', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 824, 'name': 'pressure_cooker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 825, 'name': 'printer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 826, 'name': 'projector', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 827, 'name': 'propeller', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 828, 'name': 'protective_suit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 829, 'name': 'protractor', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 830, 'name': 'prune', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 831, 'name': 'pudding', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 832, 'name': 'puffer_(fish)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 833, 'name': 'puffin', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 834, 'name': 'pug-dog', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 835, 'name': 'pumpkin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 836, 'name': 'puncher', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 837, 'name': 'puppet', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 838, 'name': 'puzzle', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 839, 'name': 'quesadilla', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 840, 'name': 'quiche', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 841, 'name': 'quilt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 842, 'name': 'quince', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 843, 'name': 'rabbit', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 844, 'name': 'raccoon', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 845, 'name': 'race_car', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 846, 'name': 'radar', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 847, 'name': 'radiator', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 848, 'name': 'radio_receiver', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 849, 'name': 'raft', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 850, 'name': 'rag_doll', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 851, 'name': 'raincoat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 852, 'name': 'raisins', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 853, 'name': 'ramen', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 854, 'name': 'rat', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 855, 'name': 'razor', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 856, 'name': 'razorblade', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 857, 'name': 'rearview_mirror', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 858, 'name': 'record_player', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 859, 'name': 'reflector', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 860, 'name': 'refrigerator', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 861, 'name': 'relay_baton', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 862, 'name': 'remote_control', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 863, 'name': 'remote_control_car', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 864, 'name': 'rhinoceros', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 865, 'name': 'rice_cooker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 866, 'name': 'rickshaw', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 867, 'name': 'rifle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 868, 'name': 'ring', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 869, 'name': 'road_map', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 870, 'name': 'roadblock', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 871, 'name': 'roast_duck', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 872, 'name': 'robe', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 873, 'name': 'rocket', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 874, 'name': 'roller_skate', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 875, 'name': 'rollerblade', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 876, 'name': 'rolling_pin', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 877, 'name': 'router_(computer_equipment)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 878, 'name': 'rowboat', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 879, 'name': 'ruler', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 880, 'name': 'runner_(carpet)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 881, 'name': 'safety_hammer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 882, 'name': 'safety_pin', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 883, 'name': 'sail', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 884, 'name': 'sailboat', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 885, 'name': 'salad_plate', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 886, 'name': 'salami', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 887, 'name': 'salmon_(fish)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 888, 'name': 'salmon_(food)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 889, 'name': 'sandal_(type_of_shoe)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 890, 'name': 'sandbag', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 891, 'name': 'sandpaper', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 892, 'name': 'sandwich', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 893, 'name': 'sardine', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 894, 'name': 'satchel', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 895, 'name': 'saucepan', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 896, 'name': 'sausage', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 897, 'name': 'saxophone', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 898, 'name': 'scallops', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 899, 'name': 'scanner', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 900, 'name': 'scarecrow', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 901, 'name': 'scarf', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 902, 'name': 'school_bus', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 903, 'name': 'scissors', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 904, 'name': 'scoreboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 905, 'name': 'scorpions', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 906, 'name': 'screw', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 907, 'name': 'screwdriver', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 908, 'name': 'scrubbing_brush', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 909, 'name': 'sculpture', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 910, 'name': 'SD_card', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 911, 'name': 'sea_urchin', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 912, 'name': 'seagull', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 913, 'name': 'seahorse', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 914, 'name': 'seal', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 915, 'name': 'seaplane', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 916, 'name': 'seashell', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 917, 'name': 'seaweed', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 918, 'name': 'sedan', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 919, 'name': 'selfie_stick', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 920, 'name': 'sewing_machine', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 921, 'name': 'shampoo', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 922, 'name': 'shark', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 923, 'name': 'sharpie', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 924, 'name': 'shaver_(electric)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 925, 'name': 'shawl', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 926, 'name': 'shears', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 927, 'name': 'sheep', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 928, 'name': 'shelf', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 929, 'name': 'shepherd_dog', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 930, 'name': 'shield', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 931, 'name': 'shirt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 932, 'name': 'shoe', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 933, 'name': 'shoehorn', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 934, 'name': 'shoeshine', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 935, 'name': 'shopping_cart', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 936, 'name': 'short_pants', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 937, 'name': 'shot_glass', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 938, 'name': 'shot_put', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 939, 'name': 'shoulder_bag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 940, 'name': 'shovel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 941, 'name': 'shower_cap', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 942, 'name': 'shower_curtain', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 943, 'name': 'shower_head', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 944, 'name': 'shredder_(for_paper)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 945, 'name': 'shrimp', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 946, 'name': 'side_table', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 947, 'name': 'signboard', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 948, 'name': 'sink', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 949, 'name': 'skateboard', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 950, 'name': 'ski', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 951, 'name': 'ski_parka', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 952, 'name': 'ski_pole', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 953, 'name': 'skirt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 954, 'name': 'skunk', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 955, 'name': 'sled', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 956, 'name': 'sleeping_bag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 957, 'name': 'slide', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 958, 'name': 'slipper_(footwear)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 959, 'name': 'smartwatch', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 960, 'name': 'smoothie', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 961, 'name': 'snake', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 962, 'name': 'snow_leopard', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 963, 'name': 'snowboard', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 964, 'name': 'snowman', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 965, 'name': 'snowmobile', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 966, 'name': 'soap', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 967, 'name': 'soccer_ball', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 968, 'name': 'sock', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 969, 'name': 'socket', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 970, 'name': 'sofa', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 971, 'name': 'sofa_bed', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 972, 'name': 'sombrero', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 973, 'name': 'soundbar', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 974, 'name': 'soupspoon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 975, 'name': 'soya_milk', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 976, 'name': 'space_shuttle', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 977, 'name': 'sparrow', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 978, 'name': 'spatula', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 979, 'name': 'speaker_(stero_equipment)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 980, 'name': 'spear', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 981, 'name': 'spectacles', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 982, 'name': 'speed_bump', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 983, 'name': 'sphygmomanometer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 984, 'name': 'spice_rack', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 985, 'name': 'spider', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 986, 'name': 'spinach', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 987, 'name': 'sponge', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 988, 'name': 'spoon', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 989, 'name': 'sportswear', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 990, 'name': 'spotlight', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 991, 'name': 'spring_rolls', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 992, 'name': 'squash', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 993, 'name': 'squid_(food)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 994, 'name': 'squirrel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 995, 'name': 'stapler_(stapling_machine)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 996, 'name': 'starfish', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 997, 'name': 'starfruit', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 998, 'name': 'starting_blocks', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 999, 'name': 'steak_(food)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1000, 'name': 'steak_knife', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1001, 'name': 'steamroller', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1002, 'name': 'steel_drum', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1003, 'name': 'steering_wheel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1004, 'name': 'stepladder', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1005, 'name': 'stereo_(sound_system)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1006, 'name': 'stethoscope', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1007, 'name': 'sticker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1008, 'name': 'stirring_rod', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1009, 'name': 'stool', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1010, 'name': 'stop_sign', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1011, 'name': 'strap', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1012, 'name': 'straw_(for_drinking)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1013, 'name': 'strawberry', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1014, 'name': 'street_sign', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1015, 'name': 'streetlight', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1016, 'name': 'string_cheese', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1017, 'name': 'stuffed_animal', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1018, 'name': 'submarine', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1019, 'name': 'subway', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1020, 'name': 'subwoofer', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1021, 'name': 'sugar_bowl', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1022, 'name': 'sugarcane_(plant)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1023, 'name': 'suit_(clothing)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1024, 'name': 'suitcase', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1025, 'name': 'sunflower', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1026, 'name': 'sunglasses', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1027, 'name': 'surfboard', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1028, 'name': 'surveillance_cameras', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1029, 'name': 'sushi', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1030, 'name': 'suspenders', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1031, 'name': 'swallow', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1032, 'name': 'swan', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1033, 'name': 'sweater', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1034, 'name': 'sweatshirt', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1035, 'name': 'sweet_potato', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1036, 'name': 'swim_cap', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1037, 'name': 'swim_ring', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1038, 'name': 'swimming_goggles', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1039, 'name': 'swimsuit', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1040, 'name': 'swing', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1041, 'name': 'sword', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1042, 'name': 'synthesizer', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1043, 'name': 'syringe', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1044, 'name': 'table', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1045, 'name': 'table_lamp', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1046, 'name': 'table-tennis_table', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1047, 'name': 'tablecloth', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1048, 'name': 'tachometer', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1049, 'name': 'tag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1050, 'name': 'taillight', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1051, 'name': 'tambourine', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1052, 'name': 'tangerine', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1053, 'name': 'tank_top_(clothing)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1054, 'name': 'tape_(sticky_cloth_or_paper)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1055, 'name': 'tape_measure', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1056, 'name': 'tapestry', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1057, 'name': 'target', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1058, 'name': 'tarp', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1059, 'name': 'tassel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1060, 'name': 'tea_bag', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1061, 'name': 'teakettle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1062, 'name': 'teapot', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1063, 'name': 'telephone', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1064, 'name': 'telephone_pole', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1065, 'name': 'telephoto_lens', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1066, 'name': 'television_set', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1067, 'name': 'tennis_ball', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1068, 'name': 'tennis_racket', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1069, 'name': 'tent', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1070, 'name': 'tequila', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1071, 'name': 'termites', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1072, 'name': 'test_tube', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1073, 'name': 'test_tube_holder', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1074, 'name': 'thermometer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1075, 'name': 'thermostat', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1076, 'name': 'thimble', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1077, 'name': 'thumbtack', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1078, 'name': 'tiger', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1079, 'name': 'tights_(clothing)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1080, 'name': 'timer', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1081, 'name': 'tinfoil', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1082, 'name': 'tissue_paper', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1083, 'name': 'toast_(food)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1084, 'name': 'toaster', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1085, 'name': 'tobacco_pipe', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1086, 'name': 'toilet', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1087, 'name': 'toilet_tissue', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1088, 'name': 'tomato', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1089, 'name': 'tongs', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1090, 'name': 'toolbox', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1091, 'name': 'toothbrush', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1092, 'name': 'toothpaste', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1093, 'name': 'toothpick', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1094, 'name': 'tow_truck', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1095, 'name': 'towel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1096, 'name': 'towel_rack', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1097, 'name': 'toy', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1098, 'name': 'tractor_(farm_equipment)', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1099, 'name': 'traffic_light', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1100, 'name': 'trailer_truck', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1101, 'name': 'train_(railroad_vehicle)', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1102, 'name': 'trampoline', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1103, 'name': 'trash_can', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1104, 'name': 'travel_pillow', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1105, 'name': 'tray', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1106, 'name': 'trench_coat', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1107, 'name': 'triangle_(musical_instrument)', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1108, 'name': 'tricycle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1109, 'name': 'tripod', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1110, 'name': 'trombone', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1111, 'name': 'trophy', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1112, 'name': 'trousers', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1113, 'name': 'trowel', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1114, 'name': 'truck', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1115, 'name': 'trumpet', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1116, 'name': 'trunk', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1117, 'name': 'tuba', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1118, 'name': 'tuna', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1119, 'name': 'turban', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1120, 'name': 'turnip', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1121, 'name': 'turtle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1122, 'name': 'tweezers', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1123, 'name': 'typewriter', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1124, 'name': 'umbrella', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1125, 'name': 'unicycle', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1126, 'name': 'urinal', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1127, 'name': 'vacuum_cleaner', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1128, 'name': 'vase', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1129, 'name': 'vending_machine', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1130, 'name': 'vent', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1131, 'name': 'vest', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1132, 'name': 'vinegar', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1133, 'name': 'viola', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1134, 'name': 'violin', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1135, 'name': 'virtual_reality_headset', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1136, 'name': 'visor', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1137, 'name': 'vodka', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1138, 'name': 'volleyball', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1139, 'name': 'vulture', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1140, 'name': 'waffle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1141, 'name': 'wagon_wheel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1142, 'name': 'waist_pack', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1143, 'name': 'walkie_talkie', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1144, 'name': 'walking_stick', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1145, 'name': 'walkman', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1146, 'name': 'wall_socket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1147, 'name': 'wallet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1148, 'name': 'walnut', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1149, 'name': 'walrus', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1150, 'name': 'warthog', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1151, 'name': 'washbasin', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1152, 'name': 'wasp', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1153, 'name': 'watch', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1154, 'name': 'water_bottle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1155, 'name': 'water_cooler', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1156, 'name': 'water_gun', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1157, 'name': 'water_heater', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1158, 'name': 'water_jug', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1159, 'name': 'water_ski', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1160, 'name': 'water_temperature_gauge', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1161, 'name': 'watering_can', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1162, 'name': 'watermelon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1163, 'name': 'webcam', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1164, 'name': 'weightlifting_belt', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1165, 'name': 'welding_torch', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1166, 'name': 'wet_suit', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1167, 'name': 'wheel', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1168, 'name': 'wheelchair', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1169, 'name': 'whipped_cream', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1170, 'name': 'whistle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1171, 'name': 'white_sugar', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1172, 'name': 'whiteboard', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1173, 'name': 'wig', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1174, 'name': 'wind_chime', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1175, 'name': 'windmill', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1176, 'name': 'windshield_wiper', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1177, 'name': 'wine_bottle', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1178, 'name': 'wine_bucket', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1179, 'name': 'wineglass', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1180, 'name': 'wireless_chargers', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1181, 'name': 'wok', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1182, 'name': 'wolf', 'partition': 2}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1183, 'name': 'wood_plane', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1184, 'name': 'wooden_spoon', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1185, 'name': 'woodpecker', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1186, 'name': 'wreath', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1187, 'name': 'wrench', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1188, 'name': 'wristband', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1189, 'name': 'wristlet', 'partition': 1}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1190, 'name': 'writing_brush', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1191, 'name': 'xylophone', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1192, 'name': 'xylophone_mallets', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1193, 'name': 'yo-yo', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1194, 'name': 'yoga_mat', 'partition': 3}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1195, 'name': 'zebra', 'partition': 0}, 
        {"color": [220, 20, 60], "isthing": 1, 'id': 1196, 'name': 'zucchini', 'partition': 1}
    ]

YTVIS_CATEGORIES_2019 = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "giant_panda"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "lizard"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "parrot"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "skateboard"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "sedan"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "ape"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "snake"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "monkey"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "hand"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "rabbit"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "duck"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "cow"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "fish"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "train"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "horse"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "turtle"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "bear"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "motorbike"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "leopard"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "fox"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "deer"},
    {"color": [188, 208, 182], "isthing": 1, "id": 26, "name": "owl"},
    {"color": [145, 148, 174], "isthing": 1, "id": 27, "name": "surfboard"},
    {"color": [106, 0, 228], "isthing": 1, "id": 28, "name": "airplane"},
    {"color": [0, 0, 70], "isthing": 1, "id": 29, "name": "truck"},
    {"color": [199, 100, 0], "isthing": 1, "id": 30, "name": "zebra"},
    {"color": [166, 196, 102], "isthing": 1, "id": 31, "name": "tiger"},
    {"color": [110, 76, 0], "isthing": 1, "id": 32, "name": "elephant"},
    {"color": [133, 129, 255], "isthing": 1, "id": 33, "name": "snowboard"},
    {"color": [0, 0, 192], "isthing": 1, "id": 34, "name": "boat"},
    {"color": [183, 130, 88], "isthing": 1, "id": 35, "name": "shark"},
    {"color": [130, 114, 135], "isthing": 1, "id": 36, "name": "mouse"},
    {"color": [107, 142, 35], "isthing": 1, "id": 37, "name": "frog"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "eagle"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "earless_seal"},
    {"color": [255, 208, 186], "isthing": 1, "id": 40, "name": "tennis_racket"},
]


YTVIS_CATEGORIES_2021 = [
    {"color": [106, 0, 228], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [174, 57, 255], "isthing": 1, "id": 2, "name": "bear"},
    {"color": [255, 109, 65], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 192], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [0, 0, 142], "isthing": 1, "id": 5, "name": "car"},
    {"color": [255, 77, 255], "isthing": 1, "id": 6, "name": "cat"},
    {"color": [120, 166, 157], "isthing": 1, "id": 7, "name": "cow"},
    {"color": [209, 0, 151], "isthing": 1, "id": 8, "name": "deer"},
    {"color": [0, 226, 252], "isthing": 1, "id": 9, "name": "dog"},
    {"color": [179, 0, 194], "isthing": 1, "id": 10, "name": "duck"},
    {"color": [174, 255, 243], "isthing": 1, "id": 11, "name": "earless_seal"},
    {"color": [110, 76, 0], "isthing": 1, "id": 12, "name": "elephant"},
    {"color": [73, 77, 174], "isthing": 1, "id": 13, "name": "fish"},
    {"color": [250, 170, 30], "isthing": 1, "id": 14, "name": "flying_disc"},
    {"color": [0, 125, 92], "isthing": 1, "id": 15, "name": "fox"},
    {"color": [107, 142, 35], "isthing": 1, "id": 16, "name": "frog"},
    {"color": [0, 82, 0], "isthing": 1, "id": 17, "name": "giant_panda"},
    {"color": [72, 0, 118], "isthing": 1, "id": 18, "name": "giraffe"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [255, 179, 240], "isthing": 1, "id": 20, "name": "leopard"},
    {"color": [119, 11, 32], "isthing": 1, "id": 21, "name": "lizard"},
    {"color": [0, 60, 100], "isthing": 1, "id": 22, "name": "monkey"},
    {"color": [0, 0, 230], "isthing": 1, "id": 23, "name": "motorbike"},
    {"color": [130, 114, 135], "isthing": 1, "id": 24, "name": "mouse"},
    {"color": [165, 42, 42], "isthing": 1, "id": 25, "name": "parrot"},
    {"color": [220, 20, 60], "isthing": 1, "id": 26, "name": "person"},
    {"color": [100, 170, 30], "isthing": 1, "id": 27, "name": "rabbit"},
    {"color": [183, 130, 88], "isthing": 1, "id": 28, "name": "shark"},
    {"color": [134, 134, 103], "isthing": 1, "id": 29, "name": "skateboard"},
    {"color": [5, 121, 0], "isthing": 1, "id": 30, "name": "snake"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "snowboard"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "squirrel"},
    {"color": [145, 148, 174], "isthing": 1, "id": 33, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 34, "name": "tennis_racket"},
    {"color": [166, 196, 102], "isthing": 1, "id": 35, "name": "tiger"},
    {"color": [0, 80, 100], "isthing": 1, "id": 36, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 37, "name": "truck"},
    {"color": [0, 143, 149], "isthing": 1, "id": 38, "name": "turtle"},
    {"color": [0, 228, 0], "isthing": 1, "id": 39, "name": "whale"},
    {"color": [199, 100, 0], "isthing": 1, "id": 40, "name": "zebra"},
]


OVIS_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "Person"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "Bird"},
    {"color": [119, 11, 32], "isthing": 1, "id": 3, "name": "Cat"},
    {"color": [165, 42, 42], "isthing": 1, "id": 4, "name": "Dog"},
    {"color": [134, 134, 103], "isthing": 1, "id": 5, "name": "Horse"},
    {"color": [0, 0, 142], "isthing": 1, "id": 6, "name": "Sheep"},
    {"color": [255, 109, 65], "isthing": 1, "id": 7, "name": "Cow"},
    {"color": [0, 226, 252], "isthing": 1, "id": 8, "name": "Elephant"},
    {"color": [5, 121, 0], "isthing": 1, "id": 9, "name": "Bear"},
    {"color": [0, 60, 100], "isthing": 1, "id": 10, "name": "Zebra"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "Giraffe"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "Poultry"},
    {"color": [179, 0, 194], "isthing": 1, "id": 13, "name": "Giant_panda"},
    {"color": [255, 77, 255], "isthing": 1, "id": 14, "name": "Lizard"},
    {"color": [120, 166, 157], "isthing": 1, "id": 15, "name": "Parrot"},
    {"color": [73, 77, 174], "isthing": 1, "id": 16, "name": "Monkey"},
    {"color": [0, 80, 100], "isthing": 1, "id": 17, "name": "Rabbit"},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": "Tiger"},
    {"color": [0, 143, 149], "isthing": 1, "id": 19, "name": "Fish"},
    {"color": [174, 57, 255], "isthing": 1, "id": 20, "name": "Turtle"},
    {"color": [0, 0, 230], "isthing": 1, "id": 21, "name": "Bicycle"},
    {"color": [72, 0, 118], "isthing": 1, "id": 22, "name": "Motorcycle"},
    {"color": [255, 179, 240], "isthing": 1, "id": 23, "name": "Airplane"},
    {"color": [0, 125, 92], "isthing": 1, "id": 24, "name": "Boat"},
    {"color": [209, 0, 151], "isthing": 1, "id": 25, "name": "Vehical"},
]


def _get_ytvis19_image_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2019]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2019]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_ytvis21_image_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2021]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2021]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def _get_ovis_image_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def _get_lvvis_instances_meta():
    thing_ids = [k["id"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1196, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def _get_ytvis_2019_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2019 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def _get_ytvis_2021_instances_meta():
    thing_ids = [k["id"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    thing_colors = [k["color"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    assert len(thing_ids) == 40, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in YTVIS_CATEGORIES_2021 if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def _get_ovis_instances_meta():
    thing_ids = [k["id"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 25, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in OVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret



def _get_lvvis_image_meta():
    thing_ids = [k["id"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1196, len(thing_ids)
    # Mapping from the incontiguous YTVIS category id to an id in [0, 39]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


# def _get_lvvis_image_meta():
#     thing_ids = [k["id"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
#     thing_colors = [k["color"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
#     assert len(thing_ids) == 1196, len(thing_ids)
#     # Mapping from the incontiguous category id to a contiguous id in [0, C-1]
#     thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
#     thing_classes = [k["name"] for k in LVVIS_CATEGORIES if k["isthing"] == 1]
#     ret = {
#         "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
#        "thing_classes": thing_classes,
#     "thing_colors": thing_colors,
#     }
#     return ret



def load_ytvis_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, \
    dataset_name_in_dict="ytvis19", has_mask=True, has_expression=False, sot=False):
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
        if 'image_ids' in vid_dict:
            record["image_ids"] = vid_dict["image_ids"]

        video_id = record["video_id"] = vid_dict["id"]
        if has_expression:
            record["expressions"] = vid_dict["expressions"]
            # for ref-youtube-vos and ref-davis evaluation
            if "exp_id" in vid_dict:
                record["exp_id"] = vid_dict["exp_id"]
            if "video" in vid_dict:
                record["video"] = vid_dict["video"]

        video_objs = []
        for frame_idx in range(record["length"]):
            frame_objs = []
            for anno in anno_dict_list:
                assert anno["video_id"] == video_id

                obj = {key: anno[key] for key in ann_keys if key in anno}

                _bboxes = anno.get("bboxes", None)
                _segm = anno.get("segmentations", None)
                if has_mask:
                    if not (_bboxes and _segm and _bboxes[frame_idx] and _segm[frame_idx]):
                        continue
                else:
                    if not (_bboxes and _bboxes[frame_idx]):
                        continue
                if "ori_id" in anno:
                    # for VOS inference
                    obj["ori_id"] = anno["ori_id"]
                bbox = _bboxes[frame_idx]
                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if has_mask:
                    segm = _segm[frame_idx]
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
        record["has_mask"] = has_mask

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


def register_ytvis_instances(name, metadata, json_file, image_root, dataset_name_in_dict="coco", has_mask=True, has_expression=False, sot=False):
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
    DatasetCatalog.register(name, lambda: load_ytvis_json(json_file, image_root, name, \
        dataset_name_in_dict=dataset_name_in_dict,  has_mask=has_mask, has_expression=has_expression, sot=sot))

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
