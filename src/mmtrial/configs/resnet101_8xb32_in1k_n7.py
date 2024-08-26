"""Config for ResNet-101 IM1K 8x32 model.

Tuned to detect footpath distress categories (7 classes).
"""
# ruff: noqa: C408

custom_imports = dict(
    # import is relative to where your train script is located
    imports=["tx.rand_tx", "sampler.balance_sampler", "hooks"],
    allow_failed_imports=False,
)

N_CLASS = 7
TOPK = (1, 2)
N_PER_BATCH = 70
N_PER_CLASS = 10
N_EPOCH = 10
N_WORKERS = 10
DATA_ROOT = "data/category"

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

auto_scale_lr = dict(base_batch_size=256)
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    num_classes=N_CLASS,
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
        num_stages=4,
        out_indices=(3, ),
        style="pytorch",
        type="ResNet",
    ),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type="CrossEntropyLoss"),
        num_classes=N_CLASS,
        topk=TOPK,
        type="LinearClsHead",
    ),
    neck=dict(type="GlobalAveragePooling"),
    type="ImageClassifier",
)

optim_wrapper = dict(
    optimizer=dict(lr=0.02, momentum=0.9, type="SGD", weight_decay=0.0001),
    type="OptimWrapper",
)
param_scheduler = [
    # ========== LR warm-up ==========
    # If one epoch contains 933 iterations, then after 10 epochs, the learning
    # rate will reach its maximum value of 0.05.
    dict(
        begin=0,
        by_epoch=False,
        end=9330,
        start_factor=0.01,
        end_factor=1.0,
        type="LinearLR",
    ),
    # ========== LR decay ==========
    # The learning rate will be decayed by a factor of 0.5 each time AFTER
    # the specified epoch milestones.
    # We decay the learning rate first at epoch 20, 10 epochs after the warm-up
    # phase, then decay it again every 5 epochs.
    dict(
        begin=10,
        by_epoch=True,
        end=50,
        gamma=0.2,
        milestones=[19, 24, 29, 34, 39],
        type="MultiStepLR",
    ),
]

custom_hooks = [dict(type="UnfreezeMMPretrainHook", unfreeze_epoch=30)]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(
        data_root=DATA_ROOT,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(edge="short", scale=256, type="ResizeEdge"),
            dict(crop_size=224, type="CenterCrop"),
            dict(type="PackInputs"),
        ],
        split="val",
        metainfo=METAINFO,
        type="ImageNet",
    ),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)
test_evaluator = dict(topk=TOPK, type="Accuracy")
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(edge="short", scale=256, type="ResizeEdge"),
    dict(crop_size=224, type="CenterCrop"),
    dict(type="PackInputs"),
]
train_cfg = dict(by_epoch=True, max_epochs=N_EPOCH, val_interval=1)
train_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(
        data_root=DATA_ROOT,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(scale=224, type="RandomResizedCrop"),
            dict(direction="horizontal", prob=0.5, type="RandomFlip"),
            dict(type="PackInputs"),
        ],
        split="train",
        metainfo=METAINFO,
        type="ImageNet",
    ),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(scale=224, type="RandomResizedCrop"),
    dict(type="RandTx"),  # all-in-one custom transform
    dict(type="PackInputs"),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(
        data_root=DATA_ROOT,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(edge="short", scale=256, type="ResizeEdge"),
            dict(crop_size=224, type="CenterCrop"),
            dict(type="PackInputs"),
        ],
        split="val",
        metainfo=METAINFO,
        type="ImageNet",
    ),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)
val_evaluator = dict(topk=TOPK, type="Accuracy")
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
)
