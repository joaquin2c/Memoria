# coding=utf-8
from mmfewshot.detection.datasets import DrawCocoDataset
import pickle
import argparse


data_root = '../../Data/COCODraw/data'

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

ann_cfg2=[
	dict(type='ann_file',ann_file=data_root+'/Base_test.txt')
    ]


def getDataCoco(output):
  COCOval=DrawCocoDataset(
		img_prefix=data_root,multi_pipelines=train_multi_pipelines,
		ann_cfg=ann_cfg2,classes='BASE_CLASSES')
  f = open(f"{output}/valoresDraw.pckl", 'wb')
  pickle.dump(COCOval, f)
  f.close()



if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("--output",help="Dirección de la carpeta donde se guarde")
  args,unknown =parser.parse_known_args()
  output=str(args.output)
  getDataCoco(output)

