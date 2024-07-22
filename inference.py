"""Demo script to test the model using mmdet 3.3.0.

Ref: https://github.com/open-mmlab/mmdetection/issues/10829
"""
import os
from enum import Enum

import cv2
import mmcv
import mmcv.utils
from mmdet.apis import inference_detector
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS

MDL_CFG = "faster-rcnn_r50_fpn_1x_coco.py"
MDL_CHK = "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
DEVICE = "cuda:0"
IMG_DST_DIR = "."
IMG_DEMO = "../mmdetection/demo/demo.jpg"

base_dir = os.path.abspath(os.path.dirname(__file__))
ckpt_dir = os.path.join(base_dir, "checkpoints")


class TColor(Enum):
    """Colors for terminal output."""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_env() -> None:
    """Print the environment info."""
    for _, (k, v) in enumerate(mmcv.utils.collect_env().items()):
        print(f"{TColor.OKGREEN.value}=={k}== "  # noqa: T201
              f"{TColor.ENDC.value}{v}")


model = init_detector("/".join([base_dir, MDL_CFG]),
                      "/".join([ckpt_dir, MDL_CHK]),
                      device=DEVICE)

model.cfg.visualizer.save_dir = "/".join([base_dir, IMG_DST_DIR])

visualizer = VISUALIZERS.build(model.cfg.visualizer)

img_demo = cv2.imread("/".join([base_dir, IMG_DEMO]))
res = inference_detector(model, img_demo)

visualizer.add_datasample(
    name="result",
    image=mmcv.imconvert(img_demo, "bgr", "rgb"),
    data_sample=res,
    draw_gt=False,
    pred_score_thr=0.3,
    show=False,
)
