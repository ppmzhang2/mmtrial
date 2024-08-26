"""Randomly apply one of the transformations."""
# ruff: noqa: S311

import random

import numpy as np
from mmcv.transforms import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness


@TRANSFORMS.register_module()
class RandomShift(BaseTransform):
    """Shift the image and box given shift pixels and probability.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): Probability of shifts. Defaults to 0.5.
        max_shift_px (int): The max pixels for shifting. Defaults to 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Defaults to 1.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_shift_px: int = 32,
                 filter_thr_px: int = 1) -> None:
        assert 0 <= prob <= 1
        assert max_shift_px >= 0
        self.prob = prob
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)

    @cache_randomness
    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if self._random_prob() < self.prob:
            random_shift_x = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            random_shift_y = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            new_x = max(0, random_shift_x)
            ori_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            ori_y = max(0, -random_shift_y)

            # shift img
            img = results["img"]
            new_img = np.zeros_like(img)
            img_h, img_w = img.shape[:2]
            new_h = img_h - np.abs(random_shift_y)
            new_w = img_w - np.abs(random_shift_x)
            new_img[new_y:new_y + new_h,
                    new_x:new_x + new_w] = img[ori_y:ori_y + new_h,
                                               ori_x:ori_x + new_w]
            results["img"] = new_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"max_shift_px={self.max_shift_px}, "
        repr_str += f"filter_thr_px={self.filter_thr_px})"
        return repr_str
