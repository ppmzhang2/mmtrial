"""Config for DINO with Swin Transformer Large backbone on COCO."""
# ruff: noqa: C408

METAINFO = {
    "classes": ("distress", ),
    # palette is a list of color tuples, which is used for visualization.
    "palette": [
        (220, 20, 60),
    ],
}
DATA_ROOT = "data/chch/"
ANN_DIR = "annotations"
ANN_TR = "/".join([ANN_DIR, "instances_train2.json"])
ANN_VA = "/".join([ANN_DIR, "instances_val2.json"])
IMG_DIR_TR = "train/"
IMG_DIR_VA = "val/"
DATASET_TYPE = "CocoDataset"
N_CLASSES = 1
N_EP = 36
N_WORKER_TR = 1
BATCH_SIZE_TR = 1
N_WORKER_VA = 2
BATCH_SIZE_VA = 2
N_WORKER_TE = 2
BATCH_SIZE_TE = 2
RESIZE_SCALE = (1333, 800)
PACK_DET_INPUTS_META_KEYS = (
    "img_id",
    "img_path",
    "ori_shape",
    "img_shape",
    "scale_factor",
)

CKPT_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"
TX_RAND_RESIZE_1333 = dict(
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
)
TX_RAND_RESIZE_4200 = dict(
    keep_ratio=True,
    scales=[
        (400, 4200),
        (500, 4200),
        (600, 4200),
    ],
    type="RandomChoiceResize",
)
TX_RAND_CROP = dict(
    allow_negative_crop=True,
    crop_size=(384, 600),
    crop_type="absolute_range",
    type="RandomCrop",
)
PIPELINE_TR = (
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(with_bbox=True, type="LoadAnnotations"),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        transforms=[
            [TX_RAND_RESIZE_1333],
            [TX_RAND_RESIZE_4200, TX_RAND_CROP, TX_RAND_RESIZE_1333],
        ],
        type="RandomChoice",
    ),
    dict(type="PackDetInputs"),
)
PIPELINE_VA = (
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
    dict(with_bbox=True, type="LoadAnnotations"),
    dict(meta_keys=PACK_DET_INPUTS_META_KEYS, type="PackDetInputs"),
)

auto_scale_lr = dict(base_batch_size=16)
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
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(checkpoint=CKPT_URL, type="Pretrained"),
        mlp_ratio=4,
        num_heads=[6, 12, 24, 48],
        out_indices=(0, 1, 2, 3),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type="SwinTransformer",
        window_size=12,
        with_cp=True,
    ),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type="L1Loss"),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type="FocalLoss",
            use_sigmoid=True,
        ),
        loss_iou=dict(loss_weight=2.0, type="GIoULoss"),
        num_classes=N_CLASSES,
        sync_cls_avg_factor=True,
        type="DINOHead",
    ),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        pad_size_divisor=1,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        type="DetDataPreprocessor",
    ),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
        ),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True,
    ),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5,
    ),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5),
        ),
        num_layers=6,
    ),
    neck=dict(
        act_cfg=None,
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type="GN"),
        num_outs=5,
        out_channels=256,
        type="ChannelMapper",
    ),
    num_feature_levels=5,
    num_queries=900,
    positional_encoding=dict(
        normalize=True,
        num_feats=128,
        offset=0.0,
        temperature=20,
    ),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(assigner=dict(
        match_costs=[
            dict(type="FocalLossCost", weight=2.0),
            dict(box_format="xywh", type="BBoxL1Cost", weight=5.0),
            dict(iou_mode="giou", type="IoUCost", weight=2.0),
        ],
        type="HungarianAssigner",
    )),
    type="DINO",
    with_box_refine=True,
)
num_levels = 5
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type="AdamW", weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type="OptimWrapper",
)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[27, 33],
        type="MultiStepLR",
    ),
]
pretrained = CKPT_URL
resume = False
test_cfg = dict(type="TestLoop")
test_dataloader = dict(
    batch_size=BATCH_SIZE_TE,
    dataset=dict(
        ann_file=ANN_VA,
        backend_args=None,
        data_prefix=dict(img=IMG_DIR_VA),
        data_root=DATA_ROOT,
        pipeline=list(PIPELINE_VA),
        test_mode=True,
        metainfo=METAINFO,
        type=DATASET_TYPE,
    ),
    drop_last=False,
    num_workers=N_WORKER_TE,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
test_evaluator = dict(
    ann_file=DATA_ROOT + ANN_VA,
    backend_args=None,
    format_only=False,
    metric="bbox",
    type="CocoMetric",
)
test_pipeline = list(PIPELINE_VA)
train_cfg = dict(max_epochs=N_EP, type="EpochBasedTrainLoop", val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=BATCH_SIZE_TR,
    dataset=dict(
        ann_file=ANN_TR,
        backend_args=None,
        data_prefix=dict(img=IMG_DIR_TR),
        data_root=DATA_ROOT,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=list(PIPELINE_TR),
        metainfo=METAINFO,
        type=DATASET_TYPE,
    ),
    num_workers=N_WORKER_TR,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)
train_pipeline = list(PIPELINE_TR)
val_cfg = dict(type="ValLoop")
val_dataloader = dict(
    batch_size=BATCH_SIZE_VA,
    dataset=dict(
        ann_file=ANN_VA,
        backend_args=None,
        data_prefix=dict(img=IMG_DIR_VA),
        data_root=DATA_ROOT,
        pipeline=list(PIPELINE_VA),
        test_mode=True,
        metainfo=METAINFO,
        type=DATASET_TYPE,
    ),
    drop_last=False,
    num_workers=N_WORKER_VA,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)
val_evaluator = dict(
    ann_file=DATA_ROOT + ANN_VA,
    backend_args=None,
    format_only=False,
    metric="bbox",
    type="CocoMetric",
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
