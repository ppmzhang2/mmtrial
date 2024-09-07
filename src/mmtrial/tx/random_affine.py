"""Randomly apply one of the transformations."""
# ruff: noqa: S311

import math
import random

import cv2
import numpy as np
from mmcv.transforms import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness


@TRANSFORMS.register_module()
class RandomAffine(BaseTransform):
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        max_rot_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
    """

    def __init__(  # noqa: PLR0913
            self,
            max_rot_degree: float = 10.0,
            max_translate_ratio: float = 0.1,
            scaling_ratio_range: tuple[float, float] = (0.5, 1.5),
            max_shear_degree: float = 2.0,
            border: tuple[int, int] = (0, 0),
            border_val: tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rot_degree = max_rot_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val

    @cache_randomness
    def _get_rand_homo_matrix(self, height: int, width: int) -> np.ndarray:
        """Generate random homography matrix."""
        # Rotation
        rot_degree = random.uniform(-self.max_rot_degree, self.max_rot_degree)
        rotation_matrix = self._get_rotation_matrix(rot_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (random.uniform(-self.max_translate_ratio,
                                  self.max_translate_ratio) * width)
        trans_y = (random.uniform(-self.max_translate_ratio,
                                  self.max_translate_ratio) * height)
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix

    def transform(self, results: dict) -> dict:
        """Transform image with randomly generated affine matrix."""
        img = results["img"]
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_rand_homo_matrix(height, width)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val,
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_rot_degree={self.max_rot_degree}, "
        repr_str += f"max_translate_ratio={self.max_translate_ratio}, "
        repr_str += f"scaling_ratio_range={self.scaling_ratio_range}, "
        repr_str += f"max_shear_degree={self.max_shear_degree}, "
        repr_str += f"border={self.border}, "
        repr_str += f"border_val={self.border_val}, "
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian), 0.0],
                [np.sin(radian), np.cos(radian), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [
                [scale_ratio, 0.0, 0.0],
                [0.0, scale_ratio, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array(
            [
                [1, np.tan(x_radian), 0.0],
                [np.tan(y_radian), 1, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array(
            [[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32)
        return translation_matrix
