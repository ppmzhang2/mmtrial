"""Randomly apply one of the transformations."""

import random

from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import RandomAffine
from mmdet.datasets.transforms import RandomFlip
from mmdet.registry import TRANSFORMS

__all__ = ["RandTx"]


@TRANSFORMS.register_module()
class RandTx(BaseTransform):
    """Randomly apply one transformations.

    The probabilities are:
    - No transformation: 0.5
    - RandomFlip: 0.25
    - RandomAffine: 0.25

    Args:
        prob_flip (float): Probability of applying RandomFlip. Defaults 0.5.
        prob_affine (float): Probability of applying RandomAffine.
            Defaults to 0.5.
    """

    _PROB_TX = 0.5  # Probability of transformation

    def __init__(self, prob_flip: float = 0.5, prob_affine: float = 0.5):
        self._prob_flip = prob_flip * self._PROB_TX
        self._prob_affine = prob_affine * self._PROB_TX
        self._tx_flip = RandomFlip(prob=prob_flip)
        self._tx_affine = RandomAffine()

    def transform(self, results: dict) -> dict:
        """Apply the transformation based on the defined probabilities.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """
        choice = random.uniform(0, 1)  # noqa: S311

        if choice < self._prob_flip:
            # Apply RandomFlip
            results = self._tx_flip(results)
        elif choice < self._prob_flip + self._prob_affine:
            # Apply RandomAffine
            results = self._tx_affine(results)
        else:
            # No transformation
            pass

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob_flip={self._prob_flip}, "
        repr_str += f"prob_affine={self._prob_affine})"
        return repr_str
