"""Config for ResNet-101 IM1K 8x32 model.

Tuned to detect footpath distress categories (7 classes).
"""
# ruff: noqa: C408

custom_imports = dict(
    # import is relative to where your train script is located
    imports=["tx", "sampler.balance_sampler", "hooks", "models"],
    allow_failed_imports=False,
)

N_CLASS = 7
TOPK = (1, 2)
N_PER_BATCH = 140
N_PER_CLASS = 20
N_EP = 20
N_EP_FREEZE = 15
N_WORKERS = 10
DATA_ROOT = "data/category"
RESIZE_SCALE = (224, 224)
METAINFO = {
    "classes": (
        "bump",
        "crack",
        "depression",
        "displacement",
        "pothole",
        "vegetation",
        "uneven",
    ),
}

PIPELINE_TR = (
    dict(type="LoadImageFromFile"),
    dict(scale=224, crop_ratio_range=(0.9, 1.0), type="RandomResizedCrop"),
    # dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
    dict(type="RandTx"),  # all-in-one custom transform
    # dict(direction=["horizontal", "vertical"], prob=0.5, type="RandomFlip"),
    dict(type="PackInputs"),
)
PIPELINE_VA = (
    dict(type="LoadImageFromFile"),
    dict(edge="short", scale=256, type="ResizeEdge"),
    dict(crop_size=224, type="CenterCrop"),
    dict(type="PackInputs"),
)

DATASET_TR_CFG = (
    ("data_root", DATA_ROOT),
    ("pipeline", list(PIPELINE_TR)),
    ("split", "train"),
    ("metainfo", METAINFO),
    ("type", "ImageNet"),
)
DATASET_VA_CFG = (
    ("data_root", DATA_ROOT),
    ("pipeline", list(PIPELINE_VA)),
    ("split", "val"),
    ("metainfo", METAINFO),
    ("type", "ImageNet"),
)

auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    num_classes=N_CLASS,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
dataset_type = "ImageNet"
default_hooks = dict(
    checkpoint=dict(interval=1, type="CheckpointHook"),
    logger=dict(interval=100, type="LoggerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(enable=False, type="VisualizationHook"),
)
default_scope = "mmpretrain"
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend="nccl"),
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)
load_from = None
log_level = "INFO"
model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=-1,
        init_cfg=dict(checkpoint="torchvision://resnet101", type="Pretrained"),
        norm_cfg=dict(requires_grad=True, type="BN"),  # TODO: GN
        norm_eval=True,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        type="ResNet",
    ),
    data_preprocessor=data_preprocessor,
    head=dict(
        num_classes=N_CLASS,
        in_channels=(256, 512, 1024, 2048),
        loss=dict(loss_weight=1.0, type="CrossEntropyLoss"),
        topk=TOPK,
        type="LinearFpnClsHead",
    ),
    type="ImageClassifier",
)

optim_wrapper = dict(
    optimizer=dict(lr=0.0001, momentum=0.9, type="SGD", weight_decay=0.0001),
    type="OptimWrapper",
)
param_scheduler = [
    # ========== LR warm-up ==========
    # After 5 epochs, the learning rate will reach the base learning rate
    # (0.0001) from the initial one (0.00001)
    dict(
        begin=0,
        by_epoch=True,
        end=4,
        start_factor=0.1,
        end_factor=1.0,
        type="LinearLR",
    ),
    # ========== LR decay ==========
    # The learning rate will be decayed by a factor of 0.2 each time AFTER
    # the specified epoch milestones.
    dict(
        begin=0,
        by_epoch=True,
        end=N_EP,
        gamma=0.5,
        milestones=[6, 9, 12, 15],
        type="MultiStepLR",
    ),
]

# # Unfreeze the backbone after some epochs
# custom_hooks = [
#     dict(type="UnfreezeMMPretrainHook", unfreeze_epoch=N_EP_FREEZE),
# ]
randomness = dict(deterministic=False, seed=None)
resume = False

test_cfg = dict()
test_pipeline = list(PIPELINE_VA)
test_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(DATASET_VA_CFG),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)
test_evaluator = dict(topk=TOPK, type="Accuracy")

train_cfg = dict(by_epoch=True, max_epochs=N_EP, val_interval=1)
train_pipeline = list(PIPELINE_TR)
train_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(DATASET_TR_CFG),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)

val_cfg = dict()
val_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(DATASET_VA_CFG),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)
val_evaluator = dict(topk=TOPK, type="Accuracy")

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=vis_backends,
)
