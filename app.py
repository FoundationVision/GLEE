try:
    import detectron2
except:
    import os 
    os.system('pip install git+https://github.com/facebookresearch/detectron2.git')
    # os.system('git clone https://github.com/facebookresearch/detectron2.git')
    # os.system('python -m pip install -e detectron2')
    
import gradio as gr
import numpy as np
import cv2
import torch

from detectron2.config import get_cfg
from projects.GLEE.glee.models.glee_model import GLEE_Model
from projects.GLEE.glee.config import add_glee_config
# from projects.GLEE import GLEE
import torch.nn.functional as F
import torchvision
import math
from projects.GLEE.glee.data.datasets.objects365_v2 import categories as OBJ365_CATEGORIESV2


print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Tesla T4

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)



def scribble2box(img):
    if img.max()==0:
        return None, None
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    all = np.any(img,axis=2)
    R,G,B,A = img[np.where(all)[0][0],np.where(all)[1][0]].tolist()  # get color 
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return np.array([ xmin,ymin, xmax,ymax]), (R,G,B)

 
def LSJ_box_postprocess( out_bbox,  padding_size, crop_size, img_h, img_w):
    # postprocess box height and width
    boxes = box_cxcywh_to_xyxy(out_bbox)
    lsj_sclae = torch.tensor([padding_size[1], padding_size[0], padding_size[1], padding_size[0]]).to(out_bbox)
    crop_scale = torch.tensor([crop_size[1], crop_size[0], crop_size[1], crop_size[0]]).to(out_bbox)
    boxes = boxes * lsj_sclae
    boxes = boxes / crop_scale
    boxes = torch.clamp(boxes,0,1)

    scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
    scale_fct = scale_fct.to(out_bbox)
    boxes = boxes * scale_fct
    return boxes

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
                [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
                [0.700, 0.300, 0.600],[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]



coco_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
OBJ365_class_names = [cat['name'] for cat in OBJ365_CATEGORIESV2]
class_agnostic_name = ['object']

if torch.cuda.is_available():
    print('use cuda')
    device = 'cuda'
else:
    print('use cpu')
    device='cpu'

cfg_r50 = get_cfg()
add_glee_config(cfg_r50)
conf_files_r50 = 'projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml'
checkpoints_r50 = torch.load('GLEE_Lite_joint.pth') 
cfg_r50.merge_from_file(conf_files_r50)
GLEEmodel_r50 = GLEE_Model(cfg_r50, None, device, None, True).to(device)
GLEEmodel_r50.load_state_dict(checkpoints_r50, strict=False)
GLEEmodel_r50.eval()


cfg_swin = get_cfg()
add_glee_config(cfg_swin)
conf_files_swin = 'projects/GLEE/configs/images/Plus/Stage2_joint_training_CLIPteacher_SwinL.yaml'
checkpoints_swin = torch.load('GLEE_Plus_joint.pth') 
cfg_swin.merge_from_file(conf_files_swin)
GLEEmodel_swin = GLEE_Model(cfg_swin, None, device, None, True).to(device)
GLEEmodel_swin.load_state_dict(checkpoints_swin, strict=False)
GLEEmodel_swin.eval()

pixel_mean = torch.Tensor( [123.675, 116.28, 103.53]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.12, 57.375]).to(device).view(3, 1, 1)
normalizer = lambda x: (x - pixel_mean) / pixel_std
inference_size = 800
inference_type = 'resize_shot'  # or LSJ 
size_divisibility = 32

FONT_SCALE = 1.5e-3
THICKNESS_SCALE = 1e-3
TEXT_Y_OFFSET_SCALE = 1e-2 


if inference_type != 'LSJ':
    resizer = torchvision.transforms.Resize(inference_size)


def segment_image(img,prompt_mode, categoryname, custom_category, expressiong, results_select, num_inst_select, threshold_select, mask_image_mix_ration, model_selection):
    if model_selection == 'GLEE-Plus (SwinL)':
        GLEEmodel = GLEEmodel_swin
        print('use GLEE-Plus')
    else:
        GLEEmodel = GLEEmodel_r50
        print('use GLEE-Lite') 

    copyed_img = img['background'][:,:,:3].copy()

    ori_image = torch.as_tensor(np.ascontiguousarray( copyed_img.transpose(2, 0, 1)))
    ori_image = normalizer(ori_image.to(device))[None,]
    _,_, ori_height, ori_width = ori_image.shape

    if inference_type == 'LSJ':
        infer_image = torch.zeros(1,3,1024,1024).to(ori_image)
        infer_image[:,:,:inference_size,:inference_size] = ori_image
    else:
        resize_image = resizer(ori_image)
        image_size = torch.as_tensor((resize_image.shape[-2],resize_image.shape[-1]))
        re_size = resize_image.shape[-2:]
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            padding_size = ((image_size + (stride - 1)).div(stride, rounding_mode="floor") * stride).tolist()
            infer_image = torch.zeros(1,3,padding_size[0],padding_size[1]).to(resize_image)
            infer_image[0,:,:image_size[0],:image_size[1]] = resize_image
            # reversed_image = infer_image*pixel_std +  pixel_mean
            # reversed_image = torch.clip(reversed_image,min=0,max=255)
            # reversed_image = reversed_image[0].permute(1,2,0)
            # reversed_image = reversed_image.int().cpu().numpy().copy()
            # cv2.imwrite('test.png',reversed_image[:,:,::-1])


    if prompt_mode == 'categories' or prompt_mode == 'expression':
        if len(results_select)==0:
            results_select=['box']
        if  prompt_mode == 'categories':
            if  categoryname =="COCO-80":
                batch_category_name = coco_class_name
            elif categoryname =="OBJ365":
                batch_category_name = OBJ365_class_names
            elif categoryname =="Custom-List":
                batch_category_name = custom_category.split(',')
            else:
                batch_category_name = class_agnostic_name

            # mask_ori = torch.from_numpy(np.load('03_moto_mask.npy'))[None,]
            # mask_ori = (F.interpolate(mask_ori, (height, width), mode='bilinear') > 0).to(device)
            # prompt_list = [mask_ori[0]]
            prompt_list = []
            with torch.no_grad():
                (outputs,_,_) = GLEEmodel(infer_image, prompt_list, task="coco", batch_name_list=batch_category_name, is_train=False)
            topK_instance = max(num_inst_select,1)
        else:
            topK_instance = 1
            prompt_list = {'grounding':[expressiong]}
            with torch.no_grad():
                (outputs,_,_) = GLEEmodel(infer_image, prompt_list, task="grounding", batch_name_list=[], is_train=False)

        mask_pred = outputs['pred_masks'][0]
        mask_cls = outputs['pred_logits'][0]
        boxes_pred = outputs['pred_boxes'][0]

        scores = mask_cls.sigmoid().max(-1)[0]
        scores_per_image, topk_indices = scores.topk(topK_instance, sorted=True)
        if  prompt_mode == 'categories':
            valid = scores_per_image>threshold_select
            topk_indices = topk_indices[valid]
            scores_per_image = scores_per_image[valid]

        pred_class = mask_cls[topk_indices].max(-1)[1].tolist()
        pred_boxes = boxes_pred[topk_indices] 


        boxes = LSJ_box_postprocess(pred_boxes,padding_size,re_size, ori_height,ori_width)
        mask_pred = mask_pred[topk_indices]
        pred_masks = F.interpolate( mask_pred[None,], size=(padding_size[0], padding_size[1]), mode="bilinear", align_corners=False  )
        pred_masks = pred_masks[:,:,:re_size[0],:re_size[1]]
        pred_masks = F.interpolate( pred_masks, size=(ori_height,ori_width), mode="bilinear", align_corners=False  )
        pred_masks = (pred_masks>0).detach().cpu().numpy()[0]
        
        if 'mask' in results_select:

            zero_mask = np.zeros_like(copyed_img) 
            for nn, mask in enumerate(pred_masks):
                # mask = mask.numpy()
                mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

                lar = np.concatenate((mask*COLORS[nn%12][2], mask*COLORS[nn%12][1], mask*COLORS[nn%12][0]), axis = 2)
                zero_mask = zero_mask+ lar


            lar_valid = zero_mask>0
            masked_image = lar_valid*copyed_img
            img_n = masked_image*mask_image_mix_ration + np.clip(zero_mask,0,1)*255*(1-mask_image_mix_ration)
            max_p = img_n.max()
            img_n = 255*img_n/max_p
            ret = (~lar_valid*copyed_img)*mask_image_mix_ration + img_n
            ret = ret.astype('uint8') 
        else:
            ret = copyed_img

        if 'box' in results_select:

            line_width = max(ret.shape) /200
    
            for nn,(classid, box) in enumerate(zip(pred_class,boxes)):
                x1,y1,x2,y2 = box.long().tolist()
                RGB = (COLORS[nn%12][2]*255,COLORS[nn%12][1]*255,COLORS[nn%12][0]*255)
                cv2.rectangle(ret, (x1,y1), (x2,y2), RGB,  math.ceil(line_width) )
                if prompt_mode == 'categories' or (prompt_mode == 'expression' and 'expression' in results_select ):
                    if prompt_mode == 'categories':
                        label = ''
                        if 'name' in results_select:
                            label +=  batch_category_name[classid]  
                        if 'score' in results_select:
                            label +=  str(scores_per_image[nn].item())[:4] 
                    else:
                        label = expressiong

                    if len(label)==0:
                        continue
                    height, width, _ = ret.shape
                    FONT = cv2.FONT_HERSHEY_COMPLEX
                    label_width, label_height = cv2.getTextSize(label, FONT, min(width, height) * FONT_SCALE, math.ceil(min(width, height) * THICKNESS_SCALE))[0]

                    cv2.rectangle(ret, (x1,y1), (x1+label_width,(y1 -label_height) - int(height * TEXT_Y_OFFSET_SCALE)), RGB, -1)

                    cv2.putText(
                            ret,
                            label,
                            (x1, y1 - int(height * TEXT_Y_OFFSET_SCALE)),
                            fontFace=FONT,
                            fontScale=min(width, height) * FONT_SCALE,
                            thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                            color=(255,255,255),
                        )



        ret = ret.astype('uint8') 
        return ret


    else: #visual prompt
        topK_instance = 1
        copyed_img = img['background'][:,:,:3].copy()
        # get bbox from scribbles in layers
        bbox_list = [scribble2box(layer)  for layer in img['layers'] ]
        visual_prompt_list = []
        visual_prompt_RGB_list = []

        for mask, (box,RGB) in zip(img['layers'], bbox_list):
            if box is None:
                continue
            if prompt_mode=='box':
                fakemask = np.zeros_like(copyed_img[:,:,0])
                x1 ,y1 ,x2, y2 = box
                fakemask[ y1:y2, x1:x2  ] = 1
                fakemask = fakemask>0
            elif prompt_mode=='point':
                fakemask = np.zeros_like(copyed_img[:,:,0])
                H,W = fakemask.shape
                x1 ,y1 ,x2, y2 = box
                center_x, center_y = (x1+x2)//2, (y1+y2)//2
                fakemask[ center_y-H//40:center_y+H//40, center_x-W//40:center_x+W//40  ] = 1
                fakemask = fakemask>0
            elif prompt_mode=='scribble':
                fakemask = mask[:,:,-1]
                fakemask = fakemask>0

            fakemask = torch.from_numpy(fakemask).unsqueeze(0).to(ori_image)
            if inference_type == 'LSJ':
                infer_visual_prompt = torch.zeros(1,1024,1024).to(ori_image)
                infer_visual_prompt[:,:inference_size,:inference_size] = fakemask
            else:
                resize_fakemask = resizer(fakemask)
                if size_divisibility > 1:
                    # the last two dims are H,W, both subject to divisibility requirement
                    infer_visual_prompt = torch.zeros(1,padding_size[0],padding_size[1]).to(resize_fakemask)
                    infer_visual_prompt[:,:image_size[0],:image_size[1]] = resize_fakemask


            visual_prompt_list.append( infer_visual_prompt>0 )
            visual_prompt_RGB_list.append(RGB)


        mask_results_list = []
        for visual_prompt in visual_prompt_list:
            
            prompt_list = {'spatial':[visual_prompt]}

            with torch.no_grad():
                (outputs,_,_) = GLEEmodel(infer_image, prompt_list, task="coco", batch_name_list=['object'], is_train=False, visual_prompt_type=prompt_mode )

            mask_pred = outputs['pred_masks'][0]
            mask_cls = outputs['pred_logits'][0]
            boxes_pred = outputs['pred_boxes'][0]

            scores = mask_cls.sigmoid().max(-1)[0]
            scores_per_image, topk_indices = scores.topk(topK_instance, sorted=True)

            pred_class = mask_cls[topk_indices].max(-1)[1].tolist()
            pred_boxes = boxes_pred[topk_indices] 


            boxes = LSJ_box_postprocess(pred_boxes,padding_size,re_size, ori_height,ori_width)
            mask_pred = mask_pred[topk_indices]
            pred_masks = F.interpolate( mask_pred[None,], size=(padding_size[0], padding_size[1]), mode="bilinear", align_corners=False  )
            pred_masks = pred_masks[:,:,:re_size[0],:re_size[1]]
            pred_masks = F.interpolate( pred_masks, size=(ori_height,ori_width), mode="bilinear", align_corners=False  )
            pred_masks = (pred_masks>0).detach().cpu().numpy()[0]
            mask_results_list.append(pred_masks)

        zero_mask = np.zeros_like(copyed_img) 
        for mask,RGB in zip(mask_results_list,visual_prompt_RGB_list):
            mask = mask.reshape(mask.shape[-2], mask.shape[-1], 1)
            lar = np.concatenate((mask*RGB[0], mask*RGB[1],mask*RGB[2]), axis = 2)
            zero_mask = zero_mask+ lar
        lar_valid = zero_mask>0
        masked_image = lar_valid*copyed_img
        img_n = masked_image*mask_image_mix_ration + np.clip(zero_mask,0,255)*(1-mask_image_mix_ration)
        max_p = img_n.max()
        img_n = 255*img_n/max_p
        ret = (~lar_valid*copyed_img)*mask_image_mix_ration + img_n
        ret = ret.astype('uint8') 
        # cv2.imwrite('00020_inst.jpg', cv2.cvtColor(ret, cv2.COLOR_BGR2RGB))

        return  ret



# def get_select_coordinates(img):

#     # img{'background':  (H,W,3)
#     # 'layers': list[ (H,W,4(RGBA)) ],  draw map
#     # 'composite': (H,W,4(RGBA))}  ori_img concat drow

#     ori_img = img['background'][:,:,:3].copy()

#     # get bbox from scribbles in layers
#     bbox_list = [scribble2box(layer)  for layer in img['layers'] ]
#     for mask, (box,RGB) in zip(img['layers'], bbox_list):
#         if box is None:
#             continue
#         cv2.rectangle(ori_img, (box[0],box[1]), (box[2],box[3]),RGB, 3)    
#     return  ori_img
 
def visual_prompt_preview(img, prompt_mode):

    copyed_img = img['background'][:,:,:3].copy()

    # get bbox from scribbles in layers
    bbox_list = [scribble2box(layer)  for layer in img['layers'] ]
    zero_mask = np.zeros_like(copyed_img) 

    for mask, (box,RGB) in zip(img['layers'], bbox_list):
        if box is None:
            continue
        
        if prompt_mode=='box':
            fakemask = np.zeros_like(copyed_img[:,:,0])
            x1 ,y1 ,x2, y2 = box
            fakemask[ y1:y2, x1:x2  ] = 1
            fakemask = fakemask>0
        elif prompt_mode=='point':
            fakemask = np.zeros_like(copyed_img[:,:,0])
            H,W = fakemask.shape
            x1 ,y1 ,x2, y2 = box
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            fakemask[ center_y-H//40:center_y+H//40, center_x-W//40:center_x+W//40  ] = 1
            fakemask = fakemask>0
        else:
            fakemask = mask[:,:,-1]
            fakemask = fakemask>0

        mask = fakemask.reshape(fakemask.shape[0], fakemask.shape[1], 1)
        lar = np.concatenate((mask*RGB[0], mask*RGB[1],mask*RGB[2]), axis = 2)
        zero_mask = zero_mask+ lar


    img_n = copyed_img + np.clip(zero_mask,0,255)
    max_p = img_n.max()
    ret = 255*img_n/max_p
    ret = ret.astype('uint8') 
    return  ret



with gr.Blocks() as demo:
    gr.Markdown('# GLEE: General Object Foundation Model for Images and Videos at Scale')
    gr.Markdown('## [Paper](ArXiv) - [Project Page](https://glee-vision.github.io) - [Code](https://github.com/FoundationVision/GLEE) ')
    
    gr.Markdown(
        '**The functionality demonstration demo app of GLEE. Select a Tab for image or video tasks. Image tasks includes arbitrary vocabulary object detection&segmentation, any form of object name or object caption detection, referring expression comprehension, and interactive segmentation. Video tasks add object tracking functionality based on image tasks.**'
    )

    
    with gr.Tab("Image task"):
        with gr.Row():
            with gr.Column():
                
                img_input = gr.ImageEditor()
                model_select = gr.Dropdown(
                                    ["GLEE-Lite (R50)", "GLEE-Plus (SwinL)"], value = "GLEE-Lite (R50)" , multiselect=False, label="Model",  
                                )
                with gr.Row():
                    with gr.Column():
                        prompt_mode_select = gr.Radio(["point", "scribble", "box", "categories", "expression"], label="Prompt", value= "categories" , info="What kind of prompt do you want to use?")
                        category_select = gr.Dropdown(
                                    ["COCO-80", "OBJ365", "Custom-List", "Class-Agnostic"], value = "COCO-80" , multiselect=False, label="Categories", info="Choose an existing category list or class-agnostic" 
                                )
                        custom_category = gr.Textbox(
                            label="Custom Category",
                            info="Input custom category list, seperate by ',' ",
                            lines=1,
                            value="dog, cat, car, person",
                        )
                        input_expressiong = gr.Textbox(
                            label="Expression",
                            info="Input any description of an object in the image ",
                            lines=2,
                            value="the red car",
                        )
                    # with gr.Column(): 
                    with gr.Group():
                        
                        with gr.Accordion("Interactive segmentation usage",open=False):
                            gr.Markdown(
                                'For interactive segmentation:<br />\
                                    1.Draw points, boxes, or scribbles on the canvas for multiclass segmentation; use separate layers for different objects, adding layers with a "+" sign.<br />\
                                    2.Point mode accepts a single point only; multiple points default to the centroid, so use boxes or scribbles for larger objects.<br />\
                                    3.After drawing, click green "√" to preview the prompt visualization; the segmentation mask follows the chosen prompt colors.'
                                )
                        with gr.Accordion("Text based detection usage",open=False):
                            gr.Markdown(
                                'GLEE supports three kind of object perception methods: category list, textual description, and class-agnostic.<br />\
                                1.Select an existing category list from the "Categories" dropdown, like COCO or OBJ365, or customize your own list.<br />\
                                2.Enter arbitrary object name in "Custom Category", or choose the expression model and describe the object in "Expression Textbox" for single object detection only.<br />\
                                3.For class-agnostic mode, choose "Class-Agnostic" from the "Categories" dropdown.'
                            )
                        img_showbox = gr.Image(label="visual prompt area preview")

                        

                        
            with gr.Column():
                image_segment = gr.Image(label="detection and segmentation results")
                with gr.Accordion("Try More Visualization Options"):
                    results_select = gr.CheckboxGroup(["box", "mask", "name", "score", "expression"], value=["box", "mask", "name", "score"], label="Shown Results", info="The results shown on image")
                    num_inst_select = gr.Slider(1, 50, value=15, step=1, label="Num of topK instances for category based detection", info="Choose between 1 and 50 for better visualization")
                    threshold_select = gr.Slider(0, 1, value=0.2, label="Confidence Threshold", info="Choose threshold ")
                    mask_image_mix_ration = gr.Slider(0, 1, value=0.65, label="Image Brightness Ratio", info="Brightness between image and colored masks ")

            
             
        image_button = gr.Button("Detect & Segment")
        img_input.change(visual_prompt_preview,  inputs = [img_input,prompt_mode_select] ,  outputs = img_showbox)
        image_button.click(segment_image, inputs=[img_input, prompt_mode_select, category_select, custom_category,input_expressiong, results_select, num_inst_select, threshold_select, mask_image_mix_ration,model_select], outputs=image_segment)


    with gr.Tab("Video task"):
        with gr.Row():
            gr.Markdown(
                '# Due to computational resource limitations, support for video tasks is being processed and is expected to be available within a week.'
            )
            video_input = gr.Image()
        video_button = gr.Button("Segment&Track")
        
if __name__ == '__main__':
    demo.launch(inbrowser=True,)
