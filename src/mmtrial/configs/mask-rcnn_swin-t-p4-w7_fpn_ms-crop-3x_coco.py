"""Config for Mask R-CNN with Swin Transformer (Tiny) backbone on COCO."""
# ruff: noqa: C408

PTH_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
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
BATCH_SIZE_TR = 4
BATCH_SIZE_VA = 4
BATCH_SIZE_TE = 16
N_WORKER_TR = 4
N_WORKER_VA = 4
N_WORKER_TE = 16
RESIZE_SCALE = (800, 1333)

EVAL_CFG = (
    ("ann_file", "/".join([DATA_ROOT, ANN_VA])),
    ("backend_args", None),
    ("format_only", False),
    ("metric", "bbox"),
    # ("metric", ["bbox", "segm"]),
    ("type", "CocoMetric"),
)

PIPELINE_TR = (
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(with_bbox=True, with_mask=False, type="LoadAnnotations"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    type="RandomChoiceResize",
                ),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (400, 1333),
                        (500, 1333),
                        (600, 1333),
                    ],
                    type="RandomChoiceResize",
                ),
                dict(
                    allow_negative_crop=True,
                    crop_size=(384, 600),
                    crop_type="absolute_range",
                    type="RandomCrop",
                ),
                dict(
                    keep_ratio=True,
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    type="RandomChoiceResize",
                ),
            ],
        ],
        type="RandomChoice",
    ),
    dict(type="PackDetInputs"),
)
PIPELINE_VA = (
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
    dict(with_bbox=True, with_mask=False, type="LoadAnnotations"),
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
max_epochs = 36
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint=PTH_URL,
            type="Pretrained",
        ),
        mlp_ratio=4,
        num_heads=[3, 6, 12, 24],
        out_indices=(0, 1, 2, 3),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type="SwinTransformer",
        window_size=7,
        with_cp=False,
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        pad_mask=False,
        pad_size_divisor=32,
        std=[58.395, 57.12, 57.375],
        type="DetDataPreprocessor",
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],
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
                type="CrossEntropyLoss",
                use_sigmoid=False,
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
        # mask_head=dict(
        #     conv_out_channels=256,
        #     in_channels=256,
        #     loss_mask=dict(
        #         loss_weight=1.0,
        #         type="CrossEntropyLoss",
        #         use_mask=False,
        #     ),
        #     num_classes=N_CLASSES,
        #     num_convs=4,
        #     type="FCNMaskHead",
        # ),
        mask_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type="RoIAlign"),
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
        loss_bbox=dict(loss_weight=1.0, type="L1Loss"),
        loss_cls=dict(
            loss_weight=1.0,
            type="CrossEntropyLoss",
            use_sigmoid=True,
        ),
        type="RPNHead",
    ),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
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
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type="MaxIoUAssigner",
            ),
            debug=False,
            mask_size=28,
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
    type="MaskRCNN",
)
optim_wrapper = dict(
    optimizer=dict(
        betas=(0.9, 0.999),
        lr=0.0001,
        type="AdamW",
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(custom_keys=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
    )),
    type="OptimWrapper",
)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=10,
        start_factor=0.001,
        type="LinearLR",
    ),
    dict(
        begin=0,
        by_epoch=True,
        end=N_EP,
        gamma=0.4,
        milestones=[15, 20, 25, 30],
        type="MultiStepLR",
    ),
]
pretrained = PTH_URL
resume = False

test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=BATCH_SIZE_TE,
    dataset=dict(DATASET_VA_CFG),
    drop_last=False,
    num_workers=N_WORKER_TE,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(EVAL_CFG)
test_pipeline = list(PIPELINE_VA)

train_cfg = dict(max_epochs=N_EP, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=BATCH_SIZE_TR,
    dataset=dict(DATASET_TR_CFG),
    num_workers=N_WORKER_TR,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = list(PIPELINE_TR)

val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=BATCH_SIZE_VA,
    dataset=dict(DATASET_VA_CFG),
    drop_last=False,
    num_workers=N_WORKER_VA,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(EVAL_CFG)

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
