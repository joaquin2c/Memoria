# coding=utf-8
from mmfewshot.detection.datasets import DrawCocoDataset, FewShotCocoDataset,QueryAwareDataset,DrawSuppFewShotVOCDataset
import pickle
import argparse


data_root = '../../Data/'
data_root_query= '../../Data/COCO'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_multi_pipelines = dict(
    query=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(1000, 440), (1000, 472), (1000, 504), (1000, 536),
                       (1000, 568), (1000, 600)],
            keep_ratio=True,
            multiscale_mode='value'),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ],
    support=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', flip_ratio=0.0),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])

'''
ann_cfg_draw=[
	dict(type='ann_file',ann_file=data_root+'/Base_train.txt')
    ]

ann_cfg_coco=[
        dict(
        type='ann_file',
         ann_file=data_root_query+'/annotations/instances_train2017.json'),
    ]

'''

ann_cfg_draw=[
	dict(type='ann_file',ann_file=data_root+'/Draw/aeroplane/trainval.txt')
    ]

ann_cfg_coco=[
        dict(
        type='ann_file',
         ann_file=data_root_query+'/annotations/instances_train2017.json'),
    ]



def getDataCoco(output):
  COCODraw=DrawSuppFewShotVOCDataset(img_prefix=data_root,multi_pipelines=train_multi_pipelines,
                ann_cfg=ann_cfg_draw,classes='BASE_CLASSES_SPLIT1',instance_wise=False)

  '''
  COCODraw=DrawCocoDataset(
                img_prefix=data_root,multi_pipelines=train_multi_pipelines,
                ann_cfg=ann_cfg_draw,classes='BASE_CLASSES',instance_wise=True)
  COCO=FewShotCocoDataset(
		img_prefix=data_root_query,multi_pipelines=train_multi_pipelines,
		ann_cfg=ann_cfg_coco,classes='BASE_CLASSES',instance_wise=False)
  query=QueryAwareDataset(num_support_ways=2,num_support_shots=1,query_dataset=COCO,support_dataset=COCODraw)
  '''
  f = open(f"{output}/VOCDraw.pckl", 'wb')
  pickle.dump(COCODraw, f)
  print(f"elemento:{COCODraw.data_infos}")
  f.close()


if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("--output",help="Dirección de la carpeta donde se guarde")
  args,unknown =parser.parse_known_args()
  output=str(args.output)
  getDataCoco(output)

