"""Config for Cascade R-CNN with ResNet-101-FPN backbone on COCO dataset."""
# ruff: noqa: C408

METAINFO = {
    "classes": ("distress", ),
    # palette is a list of color tuples, which is used for visualization.
    "palette": [
        (220, 20, 60),
    ],
}
DATA_ROOT = "data/chch/"
ANN_TR = "/".join(["annotations", "instances_train2.json"])
ANN_VA = "/".join(["annotations", "instances_val2.json"])
IMG_DIR_TR = "train/"
IMG_DIR_VA = "val/"
DATASET_TYPE = "CocoDataset"
N_CLASSES = 1
N_EP = 40
BATCH_SIZE_TR = 8
BATCH_SIZE_VA = 4
BATCH_SIZE_TE = 12
N_WORKER_TR = 8
N_WORKER_VA = 4
N_WORKER_TE = 12
RESIZE_SCALE = (800, 1333)

EVAL_CFG = (
    ("ann_file", "/".join([DATA_ROOT, ANN_VA])),
    ("backend_args", None),
    ("format_only", False),
    ("metric", "bbox"),
    ("type", "CocoMetric"),
)

PIPELINE_VA = (
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(with_bbox=True, type="LoadAnnotations"),
    dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
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
)
PIPELINE_TR = (
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(with_bbox=True, type="LoadAnnotations"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        transforms=[
            [
                dict(type="RandomShift", prob=0.5, max_shift_px=32),
                dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
            ],
            [
                dict(
                    type="RandomAffine",
                    max_rotate_degree=5,
                    max_translate_ratio=0.05,
                    scaling_ratio_range=(0.9, 1.1),
                    max_shear_degree=1,
                ),
                dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
            ],
            [
                dict(
                    type="RandomGrayscale",
                    prob=0.5,
                    keep_channels=True,
                    channel_weights=[1.0, 1.0, 1.0],
                    color_format="bgr",
                ),
                dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
            ],
            [
                dict(
                    type="RandomCrop",
                    crop_size=(600, 800),
                    crop_type="absolute_range",
                    allow_negative_crop=False,
                ),
                dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
            ],
        ],
        type="RandomChoice",
    ),
    dict(type="PackDetInputs"),
)

DATASET_VA_CFG = (
    ("ann_file", ANN_VA),
    ("backend_args", None),
    ("data_prefix", dict(img=IMG_DIR_VA)),
    ("data_root", DATA_ROOT),
    ("pipeline", list(PIPELINE_VA)),
    ("test_mode", True),
    ("metainfo", METAINFO),
    ("type", DATASET_TYPE),
)
DATASET_TR_CFG = (
    ("ann_file", ANN_TR),
    ("backend_args", None),
    ("data_prefix", dict(img=IMG_DIR_TR)),
    ("data_root", DATA_ROOT),
    ("filter_cfg", dict(filter_empty_gt=True, min_size=32)),
    ("pipeline", list(PIPELINE_TR)),
    ("metainfo", METAINFO),
    ("type", DATASET_TYPE),
)

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
        depth=101,
        frozen_stages=1,
        init_cfg=dict(checkpoint="torchvision://resnet101", type="Pretrained"),
        norm_cfg=dict(requires_grad=True, type="BN"),
        norm_eval=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        pad_size_divisor=32,
        std=[58.395, 57.12, 57.375],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        in_channels=[256, 512, 1024, 2048],
        num_outs=5,
        out_channels=256,
        type="FPN",
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                    type="DeltaXYWHBBoxCoder",
                ),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type="SmoothL1Loss"),
                loss_cls=dict(
                    loss_weight=1.0,
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                ),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type="Shared2FCBBoxHead",
            ),
            dict(
                bbox_coder=dict(
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                    type="DeltaXYWHBBoxCoder",
                ),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type="SmoothL1Loss"),
                loss_cls=dict(
                    loss_weight=1.0,
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                ),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type="Shared2FCBBoxHead",
            ),
            dict(
                bbox_coder=dict(
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                    type="DeltaXYWHBBoxCoder",
                ),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type="SmoothL1Loss"),
                loss_cls=dict(
                    loss_weight=1.0,
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                ),
                num_classes=80,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type="Shared2FCBBoxHead",
            ),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type="RoIAlign"),
            type="SingleRoIExtractor",
        ),
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        type="CascadeRoIHead",
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
        loss_bbox=dict(
            beta=0.1111111111111111,
            loss_weight=3.0,
            type="SmoothL1Loss",
        ),
        loss_cls=dict(
            loss_weight=3.0,
            type="CrossEntropyLoss",
            use_sigmoid=True,
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
        rcnn=[
            dict(
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
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
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
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
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
        ],
        rpn=dict(
            allowed_border=0,
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
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type="nms"),
            nms_pre=2000,
        ),
    ),
    type="CascadeRCNN",
)

optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type="SGD", weight_decay=0.0001),
    type="OptimWrapper",
)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=8,
        start_factor=0.01,
        end_factor=1.0,
        type="LinearLR",
    ),
    dict(
        begin=0,
        by_epoch=True,
        end=N_EP,
        gamma=0.2,
        milestones=[10, 15, 20, 25, 30],
        type="MultiStepLR",
    ),
]
resume = False

test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=BATCH_SIZE_TE,
    num_workers=N_WORKER_TE,
    dataset=dict(DATASET_VA_CFG),
    drop_last=False,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(EVAL_CFG)
test_pipeline = list(PIPELINE_VA)

train_cfg = dict(max_epochs=N_EP, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=BATCH_SIZE_TR,
    num_workers=N_WORKER_TR,
    dataset=dict(DATASET_TR_CFG),
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = list(PIPELINE_TR)

val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=BATCH_SIZE_VA,
    num_workers=N_WORKER_VA,
    dataset=dict(DATASET_VA_CFG),
    drop_last=False,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(EVAL_CFG)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[dict(type="LocalVisBackend")],
)
