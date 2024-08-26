"""Linear FPN classification head."""

import torch
from mmpretrain.models.heads import ClsHead
from mmpretrain.registry import MODELS
from torch.nn import functional


@MODELS.register_module()
class LinearFpnClsHead(ClsHead):
    """Linear FPN classification head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (tuple[int]): Number of channels for each input feature
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: tuple[int],
        **kwargs,
    ):
        init_cfg = {"type": "Normal", "layer": "Linear", "std": 0.01}
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        out_channel = 256

        # 1x1 Convolutions to reduce channel dimensions
        # NOTE: use ModuleList to avoid the error of different types between
        #       input (e.g. cuda.FloatTensor) and weight (e.g. FloatTensor)
        self.lateral_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_ch, out_channel, kernel_size=1)
            for in_ch in in_channels
        ])

        # 3x3 Convolutions for the final feature maps
        self.fpn_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(256, out_channel, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        # Final classification head
        self.fc = torch.nn.Linear(out_channel, num_classes)

    def forward(self, feats: tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        laterals = [
            lateral_conv(feat) for lateral_conv, feat in zip(
                self.lateral_convs, feats, strict=False)
        ]
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
        # Apply 3x3 convolutions to the combined feature maps
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals))
        ]
        # Assuming using only the highest resolution output
        out = torch.mean(fpn_outs[0], dim=[2, 3])
        cls_score = self.fc(out)
        return cls_score
