# Tutorial for Training

GLEE has three training stages: (1) Objects365 & OpenImages pretraining (2) image-level joint training across 15 datasets (3) scale up training by integrating additional SA1B and GRIT data. Corresponding yaml files start with `Stage1`, `Stage2`, and `Stage3` respectively. 

By default, we train GLEE using 64 A100 GPUs with the batchsize of 128. For fine-tuning on video tasks or novel downstream image tasks (ODinW), we default to using eight A100 GPUs. Users interested in specific datasets or aiming to further improve performance by training on individual datasets can adjust the `DATASETS` config within the YAML configuration file.

We provide configurations for Stage 1, 2, and 3 training with three types of backbones—ResNet50, Swin-Large, and EVA02-Large—across the Lite, Plus, and Pro variants, under the [projects/GLEE/configs](../projects/GLEE/configs) folder.  For employing larger or novel backbones, it is advisable to initialize the components beyond the backbone with the pretrained weights from GLEE-Lite-joint to expedite convergence.



## Pretrained Backbone Weights

```bash
# Language Model (CLIP text encoder)
wget  -P projects/GLEE/clip_vit_base_patch32/  https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin   

# R50 (GLEE_Lite) warmup initialized weight
# The randomly initialized Transformer Decoder is difficult to converge when combined with the large vocabulary of Objects365 and OpenImages. 
# It is recommended to use the Transformer weights of MaskDINO (with region proposal capability) to initialize and accelerate convergence.

cd weights/
wget https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/converted_maskdino_r50_withoutclip.pth

# Swin Large backbone weight
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
# EVA02-Large backbone weight
wget https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_L_pt_m38m_p14to16.pt

# Convert EVA02 weights
python3 convert_eva02.py
```

Other pretrained GLEE models can be found in [MODEL_ZOO.md](MODEL_ZOO.md)





## Joint Training



To train from scratch, it is necessary to follow the sequence of stages 1, 2, and 3, executing the training scripts in order, with each stage building upon the weights from the previous one. 

For training on a single machine, you can execute the following command:

```bash
python3 projects/GLEE/train_net.py --config-file projects/GLEE/configs/images/<config_stageX.yaml> --num-gpus 8
```

Replace `<config_stageX.yaml>` with the actual configuration file for each stage: 

```
${GLEE_ROOT} 
    -- projects
        -- GLEE
        		-- configs
            		-- images
            				-- Lite
            						-- Stage1_pretrain_openimage_obj365_CLIPfrozen_R50.yaml
            						-- Stage2_joint_training_CLIPteacher_R50.yaml
            						-- Stage3_scaleup_CLIPteacher_R50.yaml
            				-- Plus
            						-- Stage1_pretrain_openimage_obj365_CLIPfrozen_SwinL.yaml
            						-- Stage2_joint_training_CLIPteacher_SwinL.yaml
            						-- Stage3_scaleup_CLIPteacher_SwinL.yaml
            				-- Pro
            						-- Stage1_pretrain_openimage_obj365_CLIPfrozen_EVA02L_LSJ1536.yaml
            						-- Stage2_joint_training_CLIPteacher_EVA02L.yaml
            						-- Stage3_scaleup_CLIPteacher_EVA02L.yaml
```



Our standard setup involves training on multiple machines (64 x A100), for which you can use the distributed training script:

```bash
python3 launch.py --nn <num_machines>  --port <PORT> --worker_rank <Global_Rank> --master_address $<MASTER_ADDRESS>  --config-file projects/STAnything/configs/<config_stageX.yaml>
```

Here, `<num_machines>` should be replaced with the number of machines you intend to use, `<MASTER_ADDRESS>` should be the IP address of node 0. `<PORT>` should be the same among multiple nodes. , and `<config.yaml>` with the configuration file for the specific stage of training.







# Finetune (Continuously Updated)

We also provide fine-tuning scripts that enable the fine-tuning of GLEE on downstream tasks such as ODinW and various of video tasks to achieve better performance. 

These will be made available  as soon as possible.