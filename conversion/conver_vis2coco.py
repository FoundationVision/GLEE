import json

###  for YTVIS19
anns = json.load(open('datasets/ytvis_2019/annotations/instances_train_sub.json','rb'))
new_images = []
new_annotations = []
vid_imgid_mapping = {}
image_id_start = 0
for video in anns['videos']:
    vid = video['id']
    vid_imgid_mapping[vid] = []
    for name in video['file_names']:
        new_images.append(
            { 'file_name': name, 'height': video['height'], 'width': video['width'], 'id': image_id_start}
        )
        vid_imgid_mapping[vid].append(image_id_start)
        image_id_start += 1
for ann in anns['annotations']:
    vid = ann['video_id']
    height = ann['height']
    width = ann['width']
    category_id = ann['category_id']
    iscrowd = ann['iscrowd']

    for idx, (seg,box,area) in enumerate(zip(ann['segmentations'], ann['bboxes'], ann['areas'] )):
        if seg is None:
            continue
        image_id = vid_imgid_mapping[vid][idx]
        new_annotations.append(
            {'segmentation': seg, 'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'bbox': box, 'category_id': category_id, 'id': len(new_annotations)}
        )

new_anns = {'info':anns['info'], 'licenses':anns['licenses'], 'categories':anns['categories'],  'images':new_images,  'annotations':new_annotations}

with open('datasets/ytvis_2019/annotations/ytvis19_cocofmt.json', 'w') as f: 
    json.dump(new_anns, f)



### for YTVIS21
anns = json.load(open('datasets/ytvis_2021/annotations/instances_train_sub.json','rb'))
new_images = []
new_annotations = []
vid_imgid_mapping = {}
image_id_start = 0
for video in anns['videos']:
    vid = video['id']
    vid_imgid_mapping[vid] = []
    for name in video['file_names']:
        new_images.append(
            { 'file_name': name, 'height': video['height'], 'width': video['width'], 'id': image_id_start}
        )
        vid_imgid_mapping[vid].append(image_id_start)
        image_id_start += 1
for ann in anns['annotations']:
    vid = ann['video_id']
    height = ann['height']
    width = ann['width']
    category_id = ann['category_id']
    iscrowd = ann['iscrowd']

    for idx, (seg,box,area) in enumerate(zip(ann['segmentations'], ann['bboxes'], ann['areas'] )):
        if seg is None:
            continue
        image_id = vid_imgid_mapping[vid][idx]
        new_annotations.append(
            {'segmentation': seg, 'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'bbox': box, 'category_id': category_id, 'id': len(new_annotations)}
        )

new_anns = {'info':anns['info'], 'licenses':anns['licenses'], 'categories':anns['categories'],  'images':new_images,  'annotations':new_annotations}

with open('datasets/ytvis_2021/annotations/ytvis19_cocofmt.json', 'w') as f: 
    json.dump(new_anns, f)




### for OVIS

anns = json.load(open('datasets/ovis/annotations_train.json','rb'))
new_images = []
new_annotations = []
vid_imgid_mapping = {}
image_id_start = 0
for video in anns['videos']:
    vid = video['id']
    vid_imgid_mapping[vid] = []
    for name in video['file_names']:
        new_images.append(
            { 'file_name': name, 'height': video['height'], 'width': video['width'], 'id': image_id_start}
        )
        vid_imgid_mapping[vid].append(image_id_start)
        image_id_start += 1

for ann in anns['annotations']:
    vid = ann['video_id']
    height = ann['height']
    width = ann['width']
    category_id = ann['category_id']
    iscrowd = ann['iscrowd']
    for idx, (seg,box,area) in enumerate(zip(ann['segmentations'], ann['bboxes'], ann['areas'] )):
        if seg is None:
            continue
        image_id = vid_imgid_mapping[vid][idx]
        new_annotations.append(
            {'segmentation': seg, 'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'bbox': box, 'category_id': category_id, 'id': len(new_annotations)}
        )

new_anns = {'info':anns['info'], 'licenses':anns['licenses'], 'categories':anns['categories'],  'images':new_images,  'annotations':new_annotations}

with open('datasets/ovis/ovis_cocofmt.json', 'w') as f:
    json.dump(new_anns, f)



