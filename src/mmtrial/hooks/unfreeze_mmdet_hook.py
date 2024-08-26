"""Unfreeze backbone network Hook for MMDetection."""

from mmdet.registry import HOOKS

from .unfreeze_base_hook import UnfreezeBaseHook


@HOOKS.register_module()
class UnfreezeMMDetHook(UnfreezeBaseHook):
    """Epoch-based unfreeze backbone network Hook for MMDetection."""

    def __init__(self, unfreeze_epoch: int = 10):
        super().__init__(unfreeze_epoch)
