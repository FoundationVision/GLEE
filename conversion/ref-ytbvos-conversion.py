import json


rvos = json.load(open('datasets/ref-youtube-vos/train.json','rb'))


# image item : 'file_name': 'COCO_train2014_000000581857.jpg', 'height': 640, 'width': 427, 'id': 1, 'expressions': ['the lady with the blue shirt', 'lady with back to us', 'blue shirt']}
# anno item  {'bbox': [103.93, 299.99, 134.22, 177.42], 'segmentation': [[223.18, 477.41, 178.25, 476.84, 167.3, 468.2, 156.93, 464.16, 151.17, 464.74, 141.38, 471.65, 132.16, 476.26, 125.25, 476.26, 126.98, 451.49, 113.73, 448.61, 103.93, 439.39, 111.42, 419.81, 136.19, 373.15, 140.8, 363.36, 169.03, 352.99, 166.72, 337.43, 174.21, 301.72, 184.01, 300.57, 200.14, 299.99, 214.54, 314.39, 215.69, 332.83, 211.08, 359.32, 224.91, 372.57, 232.97, 388.13, 238.15, 420.96, 237.0, 443.43, 224.91, 452.64, 219.14, 453.22]], 'image_id': 1, 'iscrowd': 0, 'category_id': 1, 'id': 1, 'area': 14863.615600000136}
# categories: [{'supercategory': 'object', 'id': 1, 'name': 'object'}]


new_images = []
new_annotations = []

vid_imgid_mapping = {}
image_id_start = 0
for video in rvos['videos']:
    vid = video['id']
    vid_imgid_mapping[vid] = []
    for name in video['file_names']:
        new_images.append(
            { 'file_name': name, 'height': video['height'], 'width': video['width'], 'id': image_id_start, 'expressions':video['expressions']}
        )
        vid_imgid_mapping[vid].append(image_id_start)
        image_id_start += 1


for ann in rvos['annotations']:
    
    vid = ann['video_id']
    # height = ann['height']
    # width = ann['width']
    category_id = 1 #ann['category_id']
    iscrowd = ann['iscrowd']

    for idx, (seg,box,area) in enumerate(zip(ann['segmentations'], ann['bboxes'], ann['areas'] )):
        if seg is None:
            continue
        image_id = vid_imgid_mapping[vid][idx]
        new_annotations.append(
            {'segmentation': seg, 'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'bbox': box, 'category_id': category_id, 'id': len(new_annotations)}
        )

new_anns = { 'categories':[{'supercategory': 'object', 'id': 1, 'name': 'object'}],
              'images':new_images,  
              'annotations':new_annotations}

with open('datasets/ref-youtube-vos/RVOS_refcocofmt.json', 'w') as f:
    json.dump(new_anns, f)



