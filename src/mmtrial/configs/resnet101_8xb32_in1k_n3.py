"""Config for ResNet-101 IM1K 8x32 model.

Tuned to detect footpath distress severity (3 classes).
"""
# ruff: noqa: C408

custom_imports = dict(
    # import is relative to where your train script is located
    imports=["tx", "sampler.balance_sampler", "hooks", "models"],
    allow_failed_imports=False,
)

N_CLASS = 3
TOPK = (1, 2)
N_PER_BATCH = 60
N_PER_CLASS = 20
N_EPOCH = 600
N_EPOCH_FREEZE = 200
N_WORKERS = 10
DATA_ROOT = "data/severity"
RESIZE_SCALE = (224, 224)

METAINFO = {
    "classes": ("fair", "poor", "verypoor"),
}

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
        frozen_stages=-1,  # -1 means all layers are trainable
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
    optimizer=dict(lr=0.001, momentum=0.9, type="SGD", weight_decay=0.0001),
    type="OptimWrapper",
)
param_scheduler = [
    # ========== LR warm-up ==========
    # After 100 epochs, the learning rate will reach the base learning rate
    # (0.001) from the initial one (0.0001)
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        start_factor=0.01,
        end_factor=1.0,
        type="LinearLR",
    ),
    # ========== LR decay ==========
    # The learning rate will be decayed by a factor of 0.2 each time AFTER
    # the specified epoch milestones.
    dict(
        begin=100,
        by_epoch=True,
        end=500,
        gamma=0.2,
        milestones=[120, 220, 320, 420],
        type="MultiStepLR",
    ),
]

# # We do NOT freeze the backbone
# custom_hooks = [
#     dict(type="UnfreezeMMPretrainHook", unfreeze_epoch=N_EPOCH_FREEZE),
# ]
randomness = dict(deterministic=False, seed=None)
resume = False

test_cfg = dict()
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(edge="short", scale=256, type="ResizeEdge"),
    dict(crop_size=224, type="CenterCrop"),
    dict(type="PackInputs"),
]
test_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(
        data_root=DATA_ROOT,
        pipeline=test_pipeline,
        split="val",
        metainfo=METAINFO,
        type="ImageNet",
    ),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)
test_evaluator = dict(topk=TOPK, type="Accuracy")

train_cfg = dict(by_epoch=True, max_epochs=N_EPOCH, val_interval=1)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(scale=224, crop_ratio_range=(0.9, 1.0), type="RandomResizedCrop"),
    # dict(keep_ratio=True, scale=RESIZE_SCALE, type="Resize"),
    dict(type="RandTx"),  # all-in-one custom transform
    # dict(direction=["horizontal", "vertical"], prob=0.5, type="RandomFlip"),
    dict(type="PackInputs"),
]
train_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(
        data_root=DATA_ROOT,
        pipeline=train_pipeline,
        split="train",
        metainfo=METAINFO,
        type="ImageNet",
    ),
    num_workers=N_WORKERS,
    sampler=dict(type="BalanceSampler", num_per_class=N_PER_CLASS),
)

val_cfg = dict()
val_dataloader = dict(
    batch_size=N_PER_BATCH,
    dataset=dict(
        data_root=DATA_ROOT,
        pipeline=test_pipeline,
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
