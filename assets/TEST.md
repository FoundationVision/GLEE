# Tutorial for Testing (Continuously Updated)

GLEE can be directly tested on classic detection and segmentation datasets locally. For some video datasets, the results need to be submitted to Codalab for evaluation. Additionally, certain datasets such as TAO, BURST, and OmniLabel require evaluation using additional tools. We will continue to update the evaluation tutorials for all datasets reported in the paper here.



## Detection，Instance Segmentation，REC&RES 

GLEE can directly perform evaluations on COCO, Objects365, LVIS, and the RefCOCO series based on Detectron2. Typically, the Stage 2 yaml config file can be used, with manual adjustments made for the dataset to be inferred and the selection of weights to be downloaded from the  [MODEL_ZOO.md](MODEL_ZOO.md).

To inference on COCO:

```bash
# Lite
python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml  --num-gpus 8 --eval-only  MODEL.WEIGHTS path/to/GLEE_Lite_joint.pth  DATASETS.TEST '("coco_2017_val",)'

# Plus
python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/images/Plus/Stage2_joint_training_CLIPteacher_SwinL.yaml  --num-gpus 8 --eval-only  MODEL.WEIGHTS path/to/GLEE_Plus_joint.pth DATASETS.TEST '("coco_2017_val",)'

# Pro
python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/images/Pro/Stage2_joint_training_CLIPteacher_EVA02L.yaml  --num-gpus 8 --eval-only  MODEL.WEIGHTS path/to/GLEE_Pro_joint.pth DATASETS.TEST '("coco_2017_val",)'

```

Replace `"path/to/downloaded/weights"` with the actual path to the pretrained model weights and use `"DATASETS.TEST"` to specific the dataset you wish to evaluate on.

`'("coco_2017_val",)'` can be replace by :

```bash
# Lite
python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml  --num-gpus 8 --eval-only  MODEL.WEIGHTS path/to/GLEE_Lite_joint.pth  DATASETS.TEST 
'("coco_2017_val",)'
'("lvis_v1_minival",)'
'("lvis_v1_val",)'
'("objects365_v2_val",)'
'("refcoco-unc-val",)'
'("refcoco-unc-testA",)'
'("refcoco-unc-testB",)'
'("refcocoplus-unc-val",)'
'("refcocoplus-unc-testA",)'
'("refcocoplus-unc-testB",)'
'("refcocog-umd-val",)'
'("refcocog-umd-test",)'
# Alternatively, to infer across all tasks at once:
'("coco_2017_val","lvis_v1_minival","lvis_v1_val","objects365_v2_val","refcoco-unc-val","refcoco-unc-testA","refcoco-unc-testB","refcocoplus-unc-val","refcocoplus-unc-testA","refcocoplus-unc-testB","refcocog-umd-val","refcocog-umd-test",)'
```



 

# Video Tasks (Continuously Updated)





# Omnilabel and ODinW (Continuously Updated)

