"""Config for Faster R-CNN with ResNet-50-FPN on COCO dataset (1x).

ref:
- https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html
- https://github.com/open-mmlab/mmdetection/issues/1432
"""
# ruff: noqa: C408

METAINFO = {
    "classes": ("distress", ),
    # palette is a list of color tuples, which is used for visualization.
    "palette": [
        (220, 20, 60),
    ],
}
DATA_ROOT = "data/chch/"
DATASET_TYPE = "CocoDataset"
METRIC_TYPE = "CocoMetric"
N_CLASSES = 1

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = DATA_ROOT
dataset_type = DATASET_TYPE
default_hooks = dict(
    checkpoint=dict(interval=1, type="CheckpointHook"),
    logger=dict(interval=50, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(type="DetVisualizationHook"),
)
default_scope = "mmdet"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
load_from = None
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)

model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint="torchvision://resnet50", type="Pretrained"),
        norm_cfg=dict(requires_grad=True, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type="FPN",
    ),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
                type="DeltaXYWHBBoxCoder",
            ),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type="L1Loss"),
            loss_cls=dict(
                loss_weight=1.0,
                use_sigmoid=False,
                type="CrossEntropyLoss",
            ),
            num_classes=N_CLASSES,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type="Shared2FCBBoxHead",
        ),
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        type="StandardRoIHead",
    ),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[0.5, 1.0, 2.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64],
            type="AnchorGenerator",
        ),
        bbox_coder=dict(
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            type="DeltaXYWHBBoxCoder",
        ),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=3.0, type="L1Loss"),
        loss_cls=dict(
            loss_weight=1.0,
            use_sigmoid=True,
            type="CrossEntropyLoss",
        ),
        type="RPNHead",
    ),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type="nms"),
            score_thr=0.05,
        ),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=1000,
        ),
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type="MaxIoUAssigner",
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type="RandomSampler",
            ),
        ),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type="MaxIoUAssigner",
            ),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type="RandomSampler",
            ),
        ),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=2000,
        ),
    ),
    type="FasterRCNN",
)

optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type="SGD", weight_decay=0.0001),
    type="OptimWrapper",
)
param_scheduler = [
    dict(begin=0, by_epoch=False, end=500, start_factor=0.001,
         type="LinearLR"),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[8, 11],
        type="MultiStepLR",
    ),
]
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file="annotations/instances_val2.json",
        backend_args=None,
        data_prefix=dict(img="val/"),
        data_root=DATA_ROOT,
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(keep_ratio=True, scale=(1333, 800), type="Resize"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        metainfo=METAINFO,
        type=DATASET_TYPE,
    ),
    drop_last=False,
    num_workers=16,  # suggested max number = 20
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file="data/chch/annotations/instances_val2.json",
    backend_args=None,
    format_only=False,
    metric="bbox",
    type=METRIC_TYPE,
)
test_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(keep_ratio=True, scale=(1333, 800), type="Resize"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]
train_cfg = dict(max_epochs=10, type="EpochBasedTrainLoop", val_interval=1)
train_pipeline = [
    # transformers:
    # open-mmlab/mmdetection/blob/3.x/mmdet/datasets/transforms/transforms.py
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(keep_ratio=True, scale=(1333, 800), type="Resize"),
    dict(prob=0.5, direction="horizontal", type="RandomFlip"),
    dict(
        brightness_delta=16,  # 32
        contrast_range=(0.9, 1.1),  # (0.5, 1.5)
        saturation_range=(0.9, 1.1),  # (0.5, 1.5)
        hue_delta=10,  # 18
        type="PhotoMetricDistortion",
    ),
    # dict(type="RandomAffine"),
    dict(type="PackDetInputs"),
]
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=16,
    dataset=dict(
        ann_file="annotations/instances_train2.json",
        backend_args=None,
        data_prefix=dict(img="train/"),
        data_root=DATA_ROOT,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=METAINFO,
        type=DATASET_TYPE,
    ),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file="annotations/instances_val2.json",
        backend_args=None,
        data_prefix=dict(img="val/"),
        data_root=DATA_ROOT,
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(keep_ratio=True, scale=(1333, 800), type="Resize"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                ),
                type="PackDetInputs",
            ),
        ],
        test_mode=True,
        metainfo=METAINFO,
        type=DATASET_TYPE,
    ),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file="data/chch/annotations/instances_val2.json",
    backend_args=None,
    format_only=False,
    metric="proposal",  # can use "bbox" if the metainfo is set
    type=METRIC_TYPE,
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
