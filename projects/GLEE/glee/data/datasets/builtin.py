
# Copyright (c) 2024 ByteDance. All Rights Reserved.
import os
from .refcoco import (
    register_refcoco,
    _get_refcoco_meta,
)
from .sa1b import (_get_sa1b_meta,register_sa1b)
from .uvo_image import (_get_uvo_image_meta, register_UVO_image)
from .uvo_video import (_get_uvo_dense_video_meta, register_UVO_dense_video)
from .burst_video import (_get_burst_video_meta, register_burst_video, _get_burst_image_meta)
from .tao import _get_tao_image_meta
# from .flicker import register_flicker, _get_flicker_meta
from detectron2.data.datasets.register_coco import register_coco_instances
from .open_image import _get_builtin_metadata_openimage
from .objects365_v2 import _get_builtin_metadata
from .objects365 import _get_builtin_metadata_obj365v1
from .vis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
    _get_ytvis19_image_meta,
    _get_ytvis21_image_meta,
    _get_ovis_image_meta,
    _get_lvvis_instances_meta,
    _get_lvvis_image_meta,
    )
from .odinw import _get_odinw_image_meta
from .rvos import (
    register_rytvis_instances,
    )
from .bdd100k import (
    _get_bdd_obj_det_meta,
    _get_bdd_inst_seg_meta,
    _get_bdd_obj_track_meta
)
from .VisualGenome import register_vg_instances, _get_vg_meta
from .omnilabel import register_omnilabel_instances, _get_omnilabel_meta

# ==== Predefined splits for REFCOCO datasets ===========
_PREDEFINED_SPLITS_REFCOCO = {
    # refcoco
    "refcoco-unc-train": ("coco/train2014", "annotations/refcoco-unc/instances_train.json"),
    "refcoco-unc-val": ("coco/train2014", "annotations/refcoco-unc/instances_val.json"),
    "refcoco-unc-testA": ("coco/train2014", "annotations/refcoco-unc/instances_testA.json"),
    "refcoco-unc-testB": ("coco/train2014", "annotations/refcoco-unc/instances_testB.json"),
    # refcocog
    "refcocog-umd-train": ("coco/train2014", "annotations/refcocog-umd/instances_train.json"),
    "refcocog-umd-val": ("coco/train2014", "annotations/refcocog-umd/instances_val.json"),
    "refcocog-umd-test": ("coco/train2014", "annotations/refcocog-umd/instances_test.json"),
    # "refcocog-google-val": ("coco/train2014", "annotations/refcocog-google/instances_val.json"),
    # refcoco+
    "refcocoplus-unc-train": ("coco/train2014", "annotations/refcocoplus-unc/instances_train.json"),
    "refcocoplus-unc-val": ("coco/train2014", "annotations/refcocoplus-unc/instances_val.json"),
    "refcocoplus-unc-testA": ("coco/train2014", "annotations/refcocoplus-unc/instances_testA.json"),
    "refcocoplus-unc-testB": ("coco/train2014", "annotations/refcocoplus-unc/instances_testB.json"),
    # mixed
    "refcoco-mixed": ("coco/train2014", "annotations/refcoco-mixed/instances_train.json"),
    "refcoco-mixed-filter": ("coco/train2014", "annotations/refcoco-mixed/instances_train_filter.json"),
    # ref_VOS_image_level
    "refytb-imagelevel": ("ref-youtube-vos/train/JPEGImages", "custom_annotations/RVOS_refcocofmt.json"),
}


def register_all_refcoco(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFCOCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_refcoco(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
 


# ==== Predefined splits for VisualGenome datasets ===========
_PREDEFINED_SPLITS_VG = {
    # mixed
    "vg_train": ("visual_genome/images", "visual_genome/annotations/train_from_objects.json"),
    "vg_captiontrain": ("visual_genome/images", "visual_genome/annotations/train.json"),
}


def register_all_vg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vg_instances(
            key,
            _get_vg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="vg"

        )

###  GRIT 20M

# ==== Predefined splits for VisualGenome datasets ===========
_PREDEFINED_SPLITS_GRIT20M = {
    "grit_30w": ("grit-20m/images/", "GRIT20M/grit_30w.json"),
    "grit_5m": ("grit-20m/images/", "GRIT20M/grit_5m.json"),
}


def register_all_grit(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_GRIT20M.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_vg_instances(
            key,
            _get_vg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="grit"
        )


# ==== Predefined splits for Omnilabel datasets ===========
_PREDEFINED_SPLITS_OmniLabel = {
    "omnilabel_coco": ("omnilabel/images", "omnilabel/omnilabel_coco.json"),
    "omnilabel_obj365": ("omnilabel/images", "omnilabel/omnilabel_obj365.json"),
    "omnilabel_openimages": ("omnilabel/images", "omnilabel/omnilabel_openimages.json"),
    "omnilabel_all": ("omnilabel/images", "omnilabel/omnilabel_cocofmt.json"),
}


def register_all_omnilabel(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OmniLabel.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_omnilabel_instances(
            key,
            _get_omnilabel_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="omnilabel"

        )


_PREDEFINED_SPLITS_SA1B = {
    # SA-1B
    "sa1b_500k": ("SA1B_scaleup/images/", "SA1B_scaleup/sa1b_500k.json"),
    "sa1b_1m": ("SA1B_scaleup/images/", "SA1B_scaleup/sa1b_1m.json"),
    "sa1b_2m": ("SA1B_scaleup/images/", "SA1B_scaleup/sa1b_2m.json"),
}


def register_all_sa1b(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SA1B.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_sa1b(
            key,
            _get_sa1b_meta(),
            os.path.join(root, json_file),
            os.path.join(root, image_root),
            has_mask = False
        )
 




_PREDEFINED_SPLITS_burst_image = {
    # BURST-image
    "image_bur": ("TAO/frames/val/", "TAO/burst_annotations/TAO_val_lvisformat.json"),
}

 
def register_all_BURST_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_burst_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_burst_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_bur",
            evaluator_type = 'lvis'
        )




_PREDEFINED_SPLITS_TAO_image = {
    # TAO-image
    "image_tao": ("TAO/frames/", "TAO/annotations-1.2/TAO_val_withlabel_lvisformat.json"),
}

 
def register_all_TAO_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TAO_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_tao_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_tao",
            evaluator_type = 'lvis'
        )




_PREDEFINED_SPLITS_LVVIS_image = {
    # ytvis-image
    "image_lv": ("lvvis/val/JPEGImages",
                       "lvvis/lvvis_cocofmt.json"),
}

 
def register_all_LVVIS_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVVIS_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_lvvis_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_lv",
        )


_PREDEFINED_SPLITS_VIS_image = {
    # ytvis-image
    "image_yt19": ("ytvis_2019/train/JPEGImages", "ytvis_2019/annotations/ytvis19_cocofmt.json"),
    "image_yt19_sub": ("ytvis_2019/train/JPEGImages", "subytvis/ytvis19_cocofmt_sub.json"),
}

 
def register_all_YTVIS19_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIS_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ytvis19_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_yt19"
        )

_PREDEFINED_SPLITS_VIS21_image = {
    # ytvis-image
    "image_yt21": ("ytvis_2021/train/JPEGImages", "ytvis_2021/annotations/ytvis21_cocofmt.json"),
}
 
def register_all_YTVIS21_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VIS21_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ytvis21_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_yt21"
        )

_PREDEFINED_SPLITS_OVIS_image = {
    # ytvis-image
    "image_o": ("ovis/train", "ovis/ovis_cocofmt.json"),
}
 
def register_all_OVIS_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ovis_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="image_o"
        )





_PREDEFINED_SPLITS_UVO_image = {
    # UVO-image
    "UVO_frame_train": ("UVO/uvo_videos_frames", "custom_annotations/UVO/annotations/FrameSet/UVO_frame_train_onecategory.json"),
    "UVO_frame_val": ("UVO/uvo_videos_frames", "custom_annotations/UVO/annotations/FrameSet/UVO_frame_val_onecategory.json"),
}


def register_all_UVO_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_UVO_image(
            key,
            _get_uvo_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'UVO_image'
        )
 

_PREDEFINED_SPLITS_UVO_dense_video = {
    # UVO-dense-video_with category
    "UVO_dense_video_train": ("UVO/uvo_videos_dense_frames_jpg", "UVO/annotations/VideoDenseSet/UVO_video_train_dense_objectlabel.json"),
    "UVO_dense_video_val": ("UVO/uvo_videos_dense_frames_jpg", "UVO/annotations/VideoDenseSet/UVO_video_val_dense_objectlabel.json"),
}


def register_all_UVO_dense_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO_dense_video.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_UVO_dense_video(
            key,
            _get_uvo_dense_video_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'uvo_video'
        )



_PREDEFINED_SPLITS_BURST_video = {
    # tao-video_without category  BURST benchmark
    "BURST_video_train": ("TAO/frames/train/", "TAO/burst_annotations/TAO_train_withlabel_ytvisformat.json"),
    "BURST_video_val": ("TAO/frames/val/", "TAO/burst_annotations/TAO_val_withlabel_ytvisformat.json"),
}


def register_all_BURST_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BURST_video.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_burst_video(
            key,
            _get_burst_video_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'burst'
        )




_PREDEFINED_SPLITS_TAO_video = {
    # tao-video_without category  BURST benchmark
    # "BURST_video_train": ("TAO/frames/train/", "TAO/burst_annotations/TAO_train_withlabel_ytvisformat.json"),
    "TAO_video_val": ("TAO/frames/", "TAO/TAO_annotations/validation_ytvisfmt.json"),
}


def register_all_TAO_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TAO_video.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_tao_image_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = 'tao_video'
        )



_PREDEFINED_SPLITS_OPEN_IMAGE = {
    "openimage_train": ("openimages/detection/", "open_image/openimages_v6_train_bbox_splitdir.json"),
    "openimage_val": ("openimages/detection/", "open_image/openimages_v6_val_bbox_splitdir.json"),
}


def register_all_openimage(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OPEN_IMAGE.items():
        register_coco_instances(
            key,
            _get_builtin_metadata_openimage(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="openimage"
        )




_PREDEFINED_SPLITS_OBJECTS365V2 = {
     "objects365_v2_train": ("Objects365V2/images/", "Objects365V2/annotations/zhiyuan_objv2_train_new.json"),
     "objects365_v2_masktrain": ("Objects365V2/images/", "Objects365V2/annotations/objects365_v2_train_with_mask.json"),
    "objects365_v2_val": ("Objects365V2/images/", "Objects365V2/annotations/zhiyuan_objv2_val_new.json"),
}



def register_all_obj365v2(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365V2.items():
        register_coco_instances(
            key,
            _get_builtin_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="obj365v2"
        )


_PREDEFINED_SPLITS_OBJECTS365V1 = {
    "objects365_v1_train": ("Objects365v1/train", "Objects365v1/objects365_train.json"),
    "objects365_v1_masktrain": ("Objects365v1/train", "Objects365v1/objects365_v1_train_with_mask.json"),
    "objects365_v1_val": ("Objects365v1/val/val", "Objects365v1/objects365_val.json"),
    "objects365_v1_val_mini": ("Objects365v1/val/val", "Objects365v1/objects365_val_mini.json"),
}

def register_all_obj365v1(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365V1.items():
        register_coco_instances(
            key,
            _get_builtin_metadata_obj365v1(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="obj365v1"
        )


######## video instance segmentationi


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/annotations/instances_train_sub.json"),
    "ytvis_2019_val": ("ytvis_2019/val/JPEGImages",
                       "ytvis_2019/annotations/instances_val_sub.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_dev": ("ytvis_2019/train/JPEGImages",
                       "ytvis_2019/instances_train_sub.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/annotations/instances_train_sub.json"),
    "ytvis_2021_val": ("ytvis_2021/val/JPEGImages",
                       "ytvis_2021/annotations/instances_val_sub.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
    "ytvis_2021_dev": ("ytvis_2021/train/JPEGImages",
                       "ytvis_2021/instances_train_sub.json"),
    "ytvis_2022_val_full": ("ytvis_2022/val/JPEGImages",
                        "ytvis_2022/instances.json"),
    "ytvis_2022_val_sub": ("ytvis_2022/val/JPEGImages",
                       "ytvis_2022/instances_sub.json"),
}


_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                         "ovis/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                       "ovis/annotations_valid.json"),
    "ovis_train_sub": ("ovis/train",
                         "ovis/ovis_sub_train.json"),
    "ovis_val_sub": ("ovis/train",
                       "ovis/ovis_sub_val.json"),
}



def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="ytvis19"
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="ytvis21"
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="ovis"
        )




_PREDEFINED_SPLITS_LVVIS = {
    "lvvis_train": ("lvvis/train/JPEGImages",
                         "lvvis/train_instances.json"),
    "lvvis_val": ("lvvis/val/JPEGImages",
                       "lvvis/val_instances.json"),
}

def register_all_lvvis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_lvvis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="lvvis"
        )





_PREDEFINED_SPLITS_REFYTBVOS = {
    "rvos-refcoco-mixed": ("coco/train2014", "annotations/refcoco-mixed/instances_train_video.json"),
    "rvos-refytb-train": ("ref-youtube-vos/train/JPEGImages", "ref-youtube-vos/train.json"),
    "rvos-refytb-val": ("ref-youtube-vos/valid/JPEGImages", "ref-youtube-vos/valid.json"),
}

def register_all_refytbvos_videos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REFYTBVOS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_rytvis_instances(
            key,
            _get_refcoco_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            has_expression=True
        )





# ==== Predefined splits for BDD100K object detection ===========
_PREDEFINED_SPLITS_BDD_OBJ_DET = {
    "bdd_det_train": ("bdd100k/images/100k/train", "bdd100k/labels/det_20/det_train_cocofmt.json"),
    "bdd_det_val": ("bdd100k/images/100k/val", "bdd100k/labels/det_20/det_val_cocofmt.json"),
}

# ==== Predefined splits for BDD100K instance segmentation ===========
_PREDEFINED_SPLITS_BDD_INST_SEG = {
    "bdd_inst_train": ("bdd100k/images/10k/train", "bdd100k/labels/ins_seg/polygons/ins_seg_train_cocoformat.json"),
    "bdd_inst_val": ("bdd100k/images/10k/val", "bdd100k/labels/ins_seg/polygons/ins_seg_val_cocoformat.json"),
}

# ==== Predefined splits for BDD100K box tracking ===========
_PREDEFINED_SPLITS_BDD_BOX_TRACK = {
    "bdd_box_track_train": ("bdd100k/images/track/train", "bdd100k/labels/box_track_20/box_track_train_cocofmt_uni.json"),
    "bdd_box_track_val": ("bdd100k/images/track/val", "bdd100k/labels/box_track_20/box_track_val_cocofmt_uni.json"),
}

# ==== Predefined splits for BDD100K seg tracking ===========
_PREDEFINED_SPLITS_BDD_SEG_TRACK = {
    "bdd_seg_track_train": ("bdd100k/images/seg_track_20/train", "bdd100k/labels/seg_track_20/seg_track_train_cocoformat_uni.json"),
    "bdd_seg_track_val": ("bdd100k/images/seg_track_20/val", "bdd100k/labels/seg_track_20/seg_track_val_cocoformat_uni.json"),
}


def register_all_bdd_obj_det(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_OBJ_DET.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_bdd_obj_det_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_det"
        )
 

def register_all_bdd_inst_seg(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_INST_SEG.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_bdd_inst_seg_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_inst"
        )


def register_all_bdd_box_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_BOX_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_track_box",
            has_mask = False
        )


def register_all_bdd_seg_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_SEG_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="bdd_track_seg"
        )



_PREDEFINED_SPLITS_SOT = {
    "ytbvos18_train": ("ytbvos18/train/JPEGImages", "ytbvos18/train/train.json"),
    "ytbvos18_val": ("ytbvos18/val/JPEGImages", "ytbvos18/val/val.json"),
    "mose_train": ("mose/train/JPEGImages", "mose/train/train.json"),
    "mose_val": ("mose/val/JPEGImages", "mose/val/val.json"),
}

SOT_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}] # only one class for visual grounding



_PREDEFINED_SPLITS_ODinW13_image = {
    "odinw13_AerialDrone": ("odinw/dataset/AerialMaritimeDrone/large/valid/", "odinw/dataset/AerialMaritimeDrone/large/valid/annotations_without_background.json"),
    "odinw13_Aquarium":  ("odinw/dataset/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid" , "odinw/dataset/Aquarium/Aquarium Combined.v2-raw-1024.coco/valid/annotations_without_background.json"),
    "odinw13_Rabbits":   ("odinw/dataset/CottontailRabbits/valid" , "odinw/dataset/CottontailRabbits/valid/annotations_without_background.json"),
    "odinw13_EgoHands":  ("odinw/dataset/EgoHands/generic/mini_val" , "odinw/dataset/EgoHands/generic/mini_val/annotations_without_background.json"),
    "odinw13_Mushrooms":  ("odinw/dataset/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid" , "odinw/dataset/NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/valid/annotations_without_background.json"),
    "odinw13_Packages":  ("odinw/dataset/Packages/Raw/valid" , "odinw/dataset/Packages/Raw/valid/annotations_without_background.json"),
    "odinw13_PascalVOC":  ("odinw/dataset/PascalVOC/valid" , "odinw/dataset/PascalVOC/valid/annotations_without_background.json"),
    "odinw13_Pistols":  ("odinw/dataset/pistols/export" , "odinw/dataset/pistols/export/val_annotations_without_background.json"),
    "odinw13_Pothole":  ("odinw/dataset/pothole/valid" , "odinw/dataset/pothole/valid/annotations_without_background.json"),
    "odinw13_Raccoon":  ("odinw/dataset/Raccoon/Raccoon.v2-raw.coco/valid" , "odinw/dataset/Raccoon/Raccoon.v2-raw.coco/valid/annotations_without_background.json"),
    "odinw13_Shellfish":  ("odinw/dataset/ShellfishOpenImages/raw/valid" , "odinw/dataset/ShellfishOpenImages/raw/valid/annotations_without_background.json"),
    "odinw13_Thermal":  ("odinw/dataset/thermalDogsAndPeople/valid" , "odinw/dataset/thermalDogsAndPeople/valid/annotations_without_background.json"),
    "odinw13_Vehicles":  ("odinw/dataset/VehiclesOpenImages/416x416/mini_val" , "odinw/dataset/VehiclesOpenImages/416x416/mini_val/annotations_without_background.json"),
}
 
def register_all_odinw_image(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ODinW13_image.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_odinw_image_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict = key
        )


def _get_sot_meta():
    thing_ids = [k["id"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_all_sot(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SOT.items():
        has_mask = ("coco" in key) or ("vos" in key) or ("davis" in key)
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict='ytbvos',
            has_mask=has_mask,
            sot=True
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # refcoco/g/+
    register_all_refcoco(_root)
    register_all_sa1b(_root)
    register_all_obj365v2(_root)
    register_all_obj365v1(_root)
    register_all_openimage(_root)
    register_all_vg(_root)
    register_all_grit(_root)

    #zero-shot
    register_all_omnilabel(_root)
    register_all_odinw_image(_root)


    register_all_lvvis(_root)
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ovis(_root)
    register_all_UVO_image(_root)
    register_all_UVO_dense_video(_root)
    register_all_BURST_video(_root)
    register_all_TAO_video(_root)
    register_all_refytbvos_videos(_root)

    # vis image format
    register_all_YTVIS19_image(_root)
    register_all_YTVIS21_image(_root)
    register_all_OVIS_image(_root)
    register_all_TAO_image(_root)
    register_all_BURST_image(_root)
    register_all_LVVIS_image(_root)

    # BDD100K
    register_all_bdd_obj_det(_root)
    register_all_bdd_inst_seg(_root)
    register_all_bdd_box_track(_root)
    register_all_bdd_seg_track(_root)

    # VOS/SOT
    register_all_sot(_root)
  
