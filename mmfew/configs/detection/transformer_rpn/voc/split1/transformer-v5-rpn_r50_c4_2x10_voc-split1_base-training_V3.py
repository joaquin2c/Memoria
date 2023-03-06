_base_ = [
    '../../../_base_/datasets/query_aware/base_voc_draw.py',
    '../../../_base_/schedules/schedule.py', '../../transformer-rpn_r50_c4_V3.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
num_support_ways = 2
num_support_shots = 1
data = dict(
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        query_dataset=dict(classes='BASE_CLASSES_SPLIT1'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT1')),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'),
    model_init=dict(classes='BASE_CLASSES_SPLIT1'))
optimizer = dict(
    lr=0.004,
    # lr = 0.0015,  # 3 * lr_default / 8
    momentum=0.9,
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=2.0)}))
lr_config = dict(warmup_iters=500, warmup_ratio=0.1, step=[16000])
# runner = dict(max_iters=18000)
# runner = dict(max_iters=48000)
# runner = dict(max_iters=60000)
runner = dict(max_iters=36000)
evaluation = dict(interval=6000)
checkpoint_config = dict(interval=6000)

model = dict(
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        aggregation_layer=dict(
            aggregator_cfgs=[
                dict(
                    type='CrossAttentionAggregator',
                    in_channels=1024,
                    num_layers=1,
                    num_heads=8,
                    embed_size=256,
                    forward_expansion=2,
                    pos_encoding=True,
                    dropout_prob=0.1
                )])),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    ),
)
