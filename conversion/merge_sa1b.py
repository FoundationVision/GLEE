import json
import os



annbase = json.load(open('sa_000000_joint.json','rb'))
for i in range(len(annbase['images'])):
    annbase['images'][i]['file_name'] = 'sa_000000/' + annbase['images'][i]['file_name']

for file_idx in range(1,50):
    ann_i = json.load(open('sa_000{}_joint.json'.format(str(file_idx).zfill(3)),'rb'))
    for i in range(len(ann_i['images'])):
        ann_i['images'][i]['file_name'] = 'sa_000{}/'.format(str(i).zfill(3)) + ann_i['images'][i]['file_name']
    annbase['images'] += ann_i['images']  
    annbase['annotations'] += ann_i['annotations']  

print('training images:',len(annbase['images']))
print('training annotations:',len(annbase['annotations']))

json.dump(annbase, open('sa1b_subtrain_500k.json' , 'w'))


"""

10k
training images: 11186
training annotations: 516653

20k
training images: 22372
training annotations: 1030482

50k
training images: 55930
training annotations: 2573006

100k
training images: 111860
training annotations: 5158346


200k
training images: 223720
training annotations: 10342363

500k
training images: 559300
training annotations: 25875267

"""

