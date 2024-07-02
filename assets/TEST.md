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

### Youtube-VIS, OVIS

1. Run the inference scripts:

   ```
   # YTVIS19 GLEE-Lite
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Lite/ytvis19_base.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Plus_joint.pth 
   # YTVIS19 GLEE-Plus
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Plus/ytvis19_Plus.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS /path/to/GLEE_model_zoo/GLEE_Plus_joint.pth 
   
   # ovis GLEE-Lite
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Lite/ovis_base.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Lite_joint.pth 
   # ovis GLEE-Plus
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Plus/ovis_Plus.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Plus_joint.pth 
   ```

2. Submit the results.zip to online servers.







### TAO, BURST

#### 1. Data preparation

TAO and BURST share the same video frames.

First, download the validation set zip files (2-TAO_VAL.zip, 2_AVA_HACS_VAL_e49d8f78098a8ffb3769617570a20903.zip) and unzip them from https://motchallenge.net/tao_download.php.

Then, download our preprocessed YTVIS format (COCO-like) annotation files from huggingface:

https://huggingface.co/spaces/Junfeng5/GLEE_demo/tree/main/annotations/TAO

And organize them as below:

```
${GLEE_ROOT}
    -- datasets
        -- TAO 
            --burst_annotations
            		-- TAO_val_withlabel_ytvisformat.json
            		-- val
            				-- all_classes.json
            				-- ...
            --TAO_annotations
            	 	-- validation_ytvisfmt.json
            	 	-- validation.json
            -- frames
            		-- val
            			-- ArgoVerse
            			-- ava
            			-- ...
           
```

#### 2. TAO

1. Run the inference scripts:

   ```
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Lite/TAO_Lite.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Lite_joint.pth 
   
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Plus/TAO_Plus.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Plus_joint.pth 
   ```

   

2. For TAO, we use teta as our evaluate metric (for more details, please refer to https://github.com/SysCV/tet/blob/main/teta/README.md)

3. Install teta and run evaluation:

   ```
   git clone https://github.com/SysCV/tet.git
   cd tet/teta/
   pip install -r requirements.txt
   pip install -e .
   
   # eval
   python3 scripts/run_tao.py --METRICS TETA --TRACKERS_TO_EVAL TETer --GT_FOLDER /path/to/${GLEE_ROOT}/datasets/TAO/TAO_annotations/validation.json  --TRACKER_SUB_FOLDER  /path/to/${GLEE_ROOT}/GLEE_TAO_Lite_640p/inference/results.json 
   
   ```

#### 3. BURST

1. Run the inference scripts:

   ```
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Lite/BURST_Lite.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Lite_joint.pth 
   
   python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/video/Plus/BURST_Plus.yaml  --eval-only --num-gpus 8 MODEL.WEIGHTS  /path/to/GLEE_model_zoo/GLEE_Plus_joint.pth 
   ```

2. Download eval tools from https://github.com/Ali2500/BURST-benchmark and https://github.com/JonathonLuiten/TrackEval:

   ```
   mkdir burst_tools
   cd burst_tools
   git clone https://github.com/Ali2500/BURST-benchmark.git
   git clone https://github.com/JonathonLuiten/TrackEval.git
   wget https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/annotations/convert_ytvis2tao.py
   ```

   

3. Run eval codes:

   ```
   # first convert ytvis format results to TAO/BURST results
   python3 convert_ytvis2tao.py  --results path/to/GLEE_BURST_Lite_720p/inference/results.json  --refer /path/to/${GLEE_ROOT}/datasets/TAO/burst_annotations/val/all_classes.json 
   
   cd BURST-benchmark
   export TRACKEVAL_DIR=/path/to/burst_tools/TrackEval/
   python3 burstapi/eval/run.py --pred ../converted_tao_results.json   --gt ../../burst_annotations/val/  --task class_guided
   ```

   



# Omnilabel and ODinW (Continuously Updated)

