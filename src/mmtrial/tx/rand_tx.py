"""Randomly apply one of the transformations."""

import random

from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PhotoMetricDistortion
from mmdet.datasets.transforms import RandomAffine
from mmdet.datasets.transforms import RandomCrop
from mmdet.datasets.transforms import RandomFlip
from mmdet.datasets.transforms import RandomShift
from mmdet.registry import TRANSFORMS

__all__ = ["RandTx"]


@TRANSFORMS.register_module()
class RandTx(BaseTransform):
    """Randomly apply to one batch at most one transformation.

    Args:
        p_cum_flip (float): Cumulative probability of applying RandomFlip.
            Default 0.4.
        p_cum_crop (float): Cumulative probability of applying RandomCrop.
            Default 0.5.
        p_cum_shift (float): Cumulative probability of applying RandomShift.
            Default 0.67.
        p_cum_photo (float): Cumulative probability of applying
            PhotoMetricDistortion. Defaults 0.84.
        p_cum_affine (float): Cumulative probability of applying RandomAffine.
            Default 1.0.
    """

    _P_TX = 0.70  # Probability of transformation
    _K_NONE = "none"
    _K_FLIP = "flip"
    _K_CROP = "crop"
    _K_SHIFT = "shift"
    _K_PHOTO = "photo"
    _K_AFFINE = "affine"

    def __init__(
        self,
        p_cum_flip: float = 0.40,
        p_cum_crop: float = 0.50,
        p_cum_shift: float = 0.67,
        p_cum_photo: float = 0.84,
        p_cum_affine: float = 1.00,
    ):
        # Define cumulative probabilities
        self._p_cum = {
            self._K_FLIP: p_cum_flip * self._P_TX,
            self._K_CROP: p_cum_crop * self._P_TX,
            self._K_SHIFT: p_cum_shift * self._P_TX,
            self._K_PHOTO: p_cum_photo * self._P_TX,
            self._K_AFFINE: p_cum_affine * self._P_TX,
        }
        # Define transformations
        # TODO: fine-tune the parameters
        self._tx = {
            self._K_FLIP:
            RandomFlip(prob=1.0, direction="horizontal"),
            self._K_CROP:
            RandomCrop(crop_size=(512, 512)),
            self._K_SHIFT:
            RandomShift(prob=1.0),
            self._K_PHOTO:
            PhotoMetricDistortion(
                brightness_delta=16,
                contrast_range=(0.9, 1.1),
                saturation_range=(0.9, 1.1),
                hue_delta=10,
            ),
            self._K_AFFINE:
            RandomAffine(
                max_rotate_degree=5.0,
                max_translate_ratio=0.05,
                scaling_ratio_range=(0.9, 1.1),
                max_shear_degree=1.0,
            ),
        }

    def _select_tx(self, rand: float) -> str:
        for name, p_cum in self._p_cum.items():
            if rand < p_cum:
                return name
        return self._K_NONE

    def transform(self, results: dict) -> dict:
        """Apply based on defined probabilities at most two transformations.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """
        rand1, rand2 = random.random(), random.random()  # noqa: S311
        key_tx1 = self._select_tx(rand1)
        key_tx2 = self._select_tx(rand2)

        if key_tx1 == self._K_NONE and key_tx2 == self._K_NONE:
            return results
        if key_tx1 == self._K_NONE:
            results = self._tx[key_tx2](results)
        if key_tx2 == self._K_NONE:
            results = self._tx[key_tx1](results)
        else:
            results = self._tx[key_tx1](results)
            results = self._tx[key_tx2](results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob_flip={self._p_cum_flip}, "
        repr_str += f"prob_crop={self._p_cum_crop}, "
        repr_str += f"prob_shift={self._p_cum_shift}, "
        repr_str += f"prob_photo={self._p_cum_photo})"
        return repr_str
