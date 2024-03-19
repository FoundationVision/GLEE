import json
import os
import torch
import pycocotools.mask as mask_util
import torch.nn.functional as F
import torchvision.ops  as ops
from tqdm import  tqdm
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser("json converter")
    parser.add_argument("--src", default="", type=str, help="the original json file")
    parser.add_argument("--des", default="", type=str, help="the processed json file")
    return parser.parse_args()


def mask_iou(mask1, mask2):
    mask1 = mask1.char()
    mask2 = mask2.char()

    intersection = (mask1[:,:,:] * mask2[:,:,:]).sum(-1).sum(-1)
    union = (mask1[:,:,:] + mask2[:,:,:] - mask1[:,:,:] * mask2[:,:,:]).sum(-1).sum(-1)
    
    iou= (intersection+1e-6) / (union+1e-6)
    ioua =  (intersection+1e-6) / (mask1[:,:,:].sum(-1).sum(-1)+1e-6)
    ioub =  (intersection+1e-6) / (mask2[:,:,:].sum(-1).sum(-1)+1e-6)
    
    return max(iou,ioua,ioub)


def batch_mask_iou(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    ) -> torch.Tensor:
    """
    Inputs:
    mask1: BxNxHxW torch.float32. Consists of [0, 1]
    mask2: BxMxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: BxNxM torch.float32. Consists of [0 - 1]
    """
    
    B, N, H, W = mask1.shape
    B, M, H, W = mask2.shape
    mask1 = mask1.view(B, N, H * W)
    mask2 = mask2.view(B, M, H * W)
    intersection = torch.matmul(mask1, mask2.swapaxes(1, 2))
    area1 = mask1.sum(dim=2).unsqueeze(1)
    area2 = mask2.sum(dim=2).unsqueeze(1)
    union = (area1.swapaxes(1, 2) + area2) - intersection

    iou_stand= (intersection+1e-6) / (union+1e-6)
    ioua =  (intersection+1e-6) / (area1+1e-6)
    ioub =  (intersection+1e-6) / (area2+1e-6)
    iou_list = torch.stack([iou_stand,ioua,ioub],dim=-1)
    iou = iou_list.max(dim=-1)[0]

    ret = torch.where(union == 0,torch.tensor(0.0, device=mask1.device), iou, )

    return ret



def mask_nms(seg_masks, scores, category_ids, nms_thr=0.5):
    ## seg_masks [num,1,H,W]
    n_samples = len(scores)
    if n_samples == 0:
        return []
    keep = [True for i in range(n_samples)]
    # seg_masks = seg_masks.sigmoid()>0.5
    
    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        # label_i = cate_labels[i]
        
        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]
            
            iou = mask_iou(mask_i,mask_j)[0]
            if iou > nms_thr:
                keep[j] = False
    return keep


def fast_mask_nms(seg_masks, scores, category_ids, nms_thr=0.5):
    ## seg_masks [num,1,H,W]
    seg_masks = seg_masks.float().permute(1,0,2,3)

    ious = batch_mask_iou(seg_masks,seg_masks)

    n_samples = len(scores)
    if n_samples == 0:
        return []
    keep = torch.tensor([True for i in range(n_samples)]).to(seg_masks.device)
    for i in range(n_samples - 1):
        valid_i = ious[0,i,i+1:] < nms_thr
        keep[i+1:] = keep[i+1:] & valid_i
        # if not keep[i]:
        #     continue
        # for j in range(i + 1, n_samples, 1):
        #     if not keep[j]:
        #         continue
        #     iou = ious[0,i,j]
        #     if iou > nms_thr:
        #         keep[j] = False
    return keep.tolist()




if __name__ == "__main__":
    args = parse_args()

    # image_names = os.listdir('{}/images'.format(args.src))
    all_names = os.listdir(args.src)
    ann_file_names = []
    for name in all_names:
        if 'json' in name:
            ann_file_names.append(name)
    ann_file_names.sort()
    inst_idx = 0 # index of the instance
    new_data = {"images": [], "annotations": [], "categories": [{"supercategory": "object","id": 1,"name": "object"}]}
    for file_name in tqdm(ann_file_names[:]):
        ann = json.load(open('{}/{}'.format(args.src, file_name),'rb'))
        image = {"file_name": ann['image']['file_name'], "height": ann['image']["height"], "width": ann['image']["width"], "id": ann['image']['image_id'],  }
        new_data["images"].append(image)
        
        # mask NMS
        score_list = []
        mask_list = []
        # box_list  = []
        for ann_i in ann['annotations']:
            segmentation = ann_i['segmentation']
            mask = mask_util.decode(segmentation)
            mask_list.append(torch.from_numpy(mask))
            score_list.append(ann_i['area']/1000)
            # box_list.append(torch.tensor(ann_i['bbox']))
        masks = torch.stack(mask_list).cuda()
        scores = torch.tensor(score_list).cuda()
        # boxes = torch.stack(box_list)


        # perform NMS
        # valids = [False for i in range(len(scores))]
        # valids[0]=True

        sort_idx = torch.sort(scores,descending=True)[1]
        masks = masks[sort_idx]
        # boxes = boxes[sort_idx]
        scores = scores[sort_idx]
        # with torch.no_grad():
        small_masks = F.interpolate(masks.unsqueeze(1), (masks.shape[-2]//8,masks.shape[-1]//8),mode = 'nearest')
        # valids_ori = mask_nms(small_masks,scores,None,0.5)
        valids = fast_mask_nms(small_masks,scores,None,0.5)
        # assert ( torch.tensor(valids) == torch.tensor(valids_ori)).any()
        valid_idx = sort_idx[valids].tolist()
        

        per_image_new_ann = []
        for idx in valid_idx: # range(len(ann['annotations'])):#valid_idx:  
            cur_data = ann['annotations'][idx]
            anno = {"bbox":cur_data["bbox"], "segmentation":cur_data["segmentation"], "image_id":ann['image']['image_id'], \
                "iscrowd":0, "category_id":1, "id":cur_data['id'], "area": cur_data['area']}
            per_image_new_ann.append(anno)

            anno_nomask = {"bbox":cur_data["bbox"], "image_id":ann['image']['image_id'], \
                "iscrowd":0, "category_id":1, "id":cur_data['id'], "area": cur_data['area']}
            new_data["annotations"].append(anno_nomask)
        json_name = '{}/{}'.format(args.src, file_name)
        os.remove(json_name)
        ann['annotations'] = per_image_new_ann
        json.dump(ann, open(json_name, 'w'))


        

    json.dump(new_data, open('{}_joint.json'.format(args.src.split('/')[-1]), 'w'))
