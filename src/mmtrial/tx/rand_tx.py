"""Randomly apply one of the transformations."""

import random

from mmcv.transforms import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms import RandomFlip
from mmcv.transforms import RandomGrayscale

from .random_affine import RandomAffine
from .random_shift import RandomShift

__all__ = ["RandTx"]

_P_TX = 0.70  # Probability of transformation
_K_NONE = "none"
_K_FLIP = "flip"
_K_SHIFT = "shift"
_K_GRAY = "gray"
_K_AFFINE = "affine"


@TRANSFORMS.register_module()
class RandTx(BaseTransform):
    """Randomly apply to one batch at most one transformation.

    RandomCrop is not included as it may return None when no bounding box
    is included in the results.

    Args:
        p_cum_flip (float): Cumulative probability of applying RandomFlip.
            Default 0.4.
        p_cum_shift (float): Cumulative probability of applying RandomShift.
            Default 0.67.
        p_cum_photo (float): Cumulative probability of applying
            PhotoMetricDistortion. Defaults 0.84.
        p_cum_affine (float): Cumulative probability of applying RandomAffine.
            Default 1.0.
    """

    def __init__(
        self,
        p_cum_flip: float = 0.50,
        p_cum_shift: float = 0.67,
        p_cum_photo: float = 0.94,
        p_cum_affine: float = 1.00,
    ):
        # Define cumulative probabilities
        self._p_cum = {
            _K_FLIP: p_cum_flip * _P_TX,
            _K_SHIFT: p_cum_shift * _P_TX,
            _K_GRAY: p_cum_photo * _P_TX,
            _K_AFFINE: p_cum_affine * _P_TX,
        }
        # Define transformations
        # TODO: fine-tune the parameters
        self._tx = {
            _K_FLIP:
            RandomFlip(prob=1.0, direction=["horizontal", "vertical"]),
            _K_SHIFT:
            RandomShift(prob=1.0),
            _K_GRAY:
            RandomGrayscale(
                prob=1.0,
                keep_channels=True,
                channel_weights=[1.0, 1.0, 1.0],
                color_format="bgr",
            ),
            _K_AFFINE:
            RandomAffine(
                max_rot_degree=5.0,
                max_translate_ratio=0.05,
                scaling_ratio_range=(0.9, 1.1),
                max_shear_degree=1.0,
            ),
        }

    def _select_tx(self, rand: float) -> str:
        for name, p_cum in self._p_cum.items():
            if rand < p_cum:
                return name
        return _K_NONE

    def transform(self, results: dict) -> dict:
        """Apply based on defined probabilities at most two transformations.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """
        x1, x2 = sorted([random.random() for _ in range(2)])  # noqa: S311
        key_tx1 = self._select_tx(x1)
        key_tx2 = self._select_tx(x2)

        # 1. no transformation
        #    since x1 < x2, if TX1 is none, TX2 is also none
        if key_tx1 == _K_NONE:
            pass
        # 2. TX1 only (TX2 is none)
        # 3. same transformations (TX1 == TX2), apply once (TX1)
        elif key_tx2 in (_K_NONE, key_tx1):
            results = self._tx[key_tx1](results)
        # 4. two different transformations
        else:
            results = self._tx[key_tx1](results)
            results = self._tx[key_tx2](results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob_flip={self._p_cum_flip}, "
        repr_str += f"prob_shift={self._p_cum_shift}, "
        repr_str += f"prob_photo={self._p_cum_photo},"
        repr_str += f"prob_affine={self._p_cum_affine})"
        return repr_str
