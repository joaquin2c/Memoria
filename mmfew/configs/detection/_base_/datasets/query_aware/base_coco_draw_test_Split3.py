# dataset settings
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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomFlip'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

SPLIT="Split3"
# classes splits are predefined in FewShotVOCDataset
data_root = '../../../Data/COCODraw/data3'
data_root_query= '../../../Data/COCO'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='QueryAwareDataset',
        num_support_ways=2,
        num_support_shots=1,
        save_dataset=False,
        query_dataset=dict(
            type='FewShotCocoDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root_query+'/annotations/instances_train2017.json'),
                 ],
            img_prefix=data_root_query,
            multi_pipelines=train_multi_pipelines,
            classes='ALL_CLASSES',
            instance_wise=False,
            min_bbox_area=32 * 32,
            split=SPLIT,
            dataset_name='query_dataset'),
        support_dataset=dict(
            type='DrawCocoDataset',
            ann_cfg=[
                dict(
                     type='ann_file',
                     ann_file=data_root+'/All_train.txt'),
            ],
            img_prefix=data_root,
            multi_pipelines=train_multi_pipelines,
            classes='All_CLASSES',
            instance_wise=True,
            min_bbox_area=32 * 32,
            split=SPLIT,
            dataset_name='support_dataset')),

    val=dict(
        type='FewShotCocoDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root_query+'/annotations/instances_val2017.json'),
                 ],
        img_prefix=data_root_query,
        pipeline=test_pipeline,
        split=SPLIT,
        classes='ALL_CLASSES',
    ),
    test=dict(
        type='FewShotCocoDataset',
            ann_cfg=[
                dict(
                    type='ann_file',
                    ann_file=data_root_query+'/annotations/instances_val2017.json'),
                 ],
        img_prefix=data_root_query,
        pipeline=test_pipeline,
        split=SPLIT,
	test_mode=True,
        classes='ALL_CLASSES',
    ),
    # random sample 10 shot base instance to evaluate training
    model_init=dict(
        copy_from_train_dataset=False,
        samples_per_gpu=16,
        workers_per_gpu=1,
        type='DrawCocoDataset',
        ann_cfg=[
                dict(
                     type='ann_file',
                     ann_file=data_root+'/All_test.txt'),
            ],
        img_prefix=data_root,
        pipeline=train_multi_pipelines['support'],
        num_base_shots=100,
        num_novel_shots=100,
        instance_wise=True,
        classes='ALL_CLASSES',
        min_bbox_area=32 * 32,
        split=SPLIT,
        dataset_name='model_init'))
evaluation = dict(interval=20000, metric='mAP')
#evaluation = dict(interval=20000, metric='bbox',classwise=True)
