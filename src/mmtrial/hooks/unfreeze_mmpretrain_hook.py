"""Unfreeze backbone network Hook for MMPretrain."""

from mmpretrain.registry import HOOKS

from .unfreeze_base_hook import UnfreezeBaseHook


@HOOKS.register_module()
class UnfreezeMMPretrainHook(UnfreezeBaseHook):
    """Epoch-based unfreeze backbone network Hook for MMPretrain."""

    def __init__(self, unfreeze_epoch: int = 10):
        super().__init__(unfreeze_epoch)
