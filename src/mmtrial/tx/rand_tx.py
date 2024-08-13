"""Randomly apply one of the transformations."""

import random

from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PhotoMetricDistortion
from mmdet.datasets.transforms import RandomCrop
from mmdet.datasets.transforms import RandomFlip
from mmdet.datasets.transforms import RandomShift
from mmdet.registry import TRANSFORMS

__all__ = ["RandTx"]


@TRANSFORMS.register_module()
class RandTx(BaseTransform):
    """Randomly apply to one batch at most one transformation.

    Args:
        prob_flip (float): Probability of applying RandomFlip. Defaults 0.4.
        prob_crop (float): Probability of applying RandomCrop. Defaults 0.3.
        prob_shift (float): Probability of applying RandomShift. Defaults 0.15.
        prob_photo (float): Probability of applying PhotoMetricDistortion.
            Defaults 0.15.
    """

    _P_TX = 0.70  # Probability of transformation

    def __init__(
        self,
        prob_flip: float = 0.40,
        prob_crop: float = 0.30,
        prob_shift: float = 0.15,
        prob_photo: float = 0.15,
    ):
        # Define cumulative probabilities
        self._prob_cum_flip = prob_flip * self._P_TX
        self._prob_cum_crop = prob_crop * self._P_TX + self._prob_cum_flip
        self._prob_cum_shift = prob_shift * self._P_TX + self._prob_cum_crop
        self._prob_cum_photo = prob_photo * self._P_TX + self._prob_cum_shift
        # Define transformations
        # TODO: fine-tune the parameters
        self._tx_flip = RandomFlip(prob=1.0, direction="horizontal")
        self._tx_crop = RandomCrop(crop_size=(64, 64))
        self._tx_shift = RandomShift(prob=1.0)
        self._tx_photo = PhotoMetricDistortion(
            brightness_delta=16,
            contrast_range=(0.9, 1.1),
            saturation_range=(0.9, 1.1),
            hue_delta=10,
        )

    def transform(self, results: dict) -> dict:
        """Apply the transformation based on the defined probabilities.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """
        choice = random.random()  # noqa: S311

        if choice < self._prob_cum_flip:
            results = self._tx_flip(results)
        elif choice < self._prob_cum_crop:
            results = self._tx_crop(results)
        elif choice < self._prob_cum_shift:
            results = self._tx_shift(results)
        elif choice < self._prob_cum_photo:
            results = self._tx_photo(results)
        else:
            pass

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob_flip={self._prob_cum_flip}, "
        repr_str += f"prob_crop={self._prob_cum_crop}, "
        repr_str += f"prob_shift={self._prob_cum_shift}, "
        repr_str += f"prob_photo={self._prob_cum_photo})"
        return repr_str
