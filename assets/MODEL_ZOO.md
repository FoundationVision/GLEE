# GLEE MODEL ZOO

## Introduction
GLEE maintains state-of-the-art (SOTA) performance across multiple tasks while preserving versatility and openness, demonstrating strong generalization capabilities. Here, we provide the model weights for all three stages of GLEE: '-pretrain', '-joint', and '-scaleup'. The '-pretrain' weights refer to those pretrained on Objects365 and OpenImages, yielding effective initializations from over three million detection data. The '-joint' weights are derived from joint training on 15 datasets, where the model achieves optimal performance. The '-scaleup' weights are obtained by incorporating additional automatically annotated SA1B and GRIT data, which enhance zero-shot performance and support a richer semantic understanding. Additionally, we offer weights fine-tuned on VOS data for interactive video tracking applications.

###  Stage 1: Pretraining 

|        Name        |                            Config                            |                            Weight                            |
| :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| GLEE-Lite-pretrain |     Stage1_pretrain_openimage_obj365_CLIPfrozen_R50.yaml     | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Lite_pretrain.pth) |
| GLEE-Plus-pretrain |    Stage1_pretrain_openimage_obj365_CLIPfrozen_SwinL.yaml    | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Plus_pretrain.pth) |
| GLEE-Pro-pretrain  | Stage1_pretrain_openimage_obj365_CLIPfrozen_EVA02L_LSJ1536.yaml | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Pro_pretrain.pth) |



### Stage 2: Image-level Joint Training 

|      Name       |                    Config                     |                            Weight                            |
| :-------------: | :-------------------------------------------: | :----------------------------------------------------------: |
| GLEE-Lite-joint |  Stage2_joint_training_CLIPteacher_R50.yaml   | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Lite_joint.pth) |
| GLEE-Plus-joint |    Stage2_joint_training_CLIPteacher_SwinL    | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Plus_joint.pth) |
| GLEE-Pro-joint  | Stage2_joint_training_CLIPteacher_EVA02L.yaml | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Pro_joint.pth) |

### Stage 3: Scale-up Training

|       Name        |                 Config                 |                            Weight                            |
| :---------------: | :------------------------------------: | :----------------------------------------------------------: |
| GLEE-Lite-scaleup |  Stage3_scaleup_CLIPteacher_R50.yaml   | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Lite_scaleup.pth) |
| GLEE-Plus-scaleup |    Stage3_scaleup_CLIPteacher_SwinL    | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Plus_scaleup.pth) |
| GLEE-Pro-scaleup  | Stage3_scaleup_CLIPteacher_EVA02L.yaml | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Pro_scaleup.pth) |

 



### Single Tasks
We also provide models trained on a VOS task with ResNet-50 backbone:

|     Name      |           Config            |                            Weight                            |
| :-----------: | :-------------------------: | :----------------------------------------------------------: |
| GLEE-Lite-vos | VOS_joint_finetune_R50.yaml | [Model](https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/MODEL_ZOO/GLEE_Lite_vos.pth) |