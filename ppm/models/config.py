from dataclasses import dataclass
from typing import Optional, Union, List
import torch.nn as nn


@dataclass
class FreezeConfig:
    """
    Configures which layers/blocks to freeze in a PyTorch model.

    Args:
        layers (Optional[Union[str, int, List[Union[str, int]]]):
            - Names (str) or indices (int) of layers to freeze
            - If `None`, freezes all parameters
        module_path (Optional[str]):
            - Dot-separated path to a `ModuleList` in the model (e.g., "bert.encoder.layer")
            - Required when using integer indices to locate layers
    """

    ix_layers: Optional[Union[int, List[int]]] = None
    module_path: Optional[str] = None

    def apply(self, model: nn.Module) -> None:
        """Freezes specified parameters in the model"""
        self._freeze_all(model)

        if self.ix_layers is None:
            return

        ix_layers = (
            [self.ix_layers] if not isinstance(self.ix_layers, list) else self.ix_layers
        )
        for ix in ix_layers:
            if isinstance(ix, int):
                self._unfreeze_by_index(model, ix)
            else:
                raise TypeError(f"Invalid layer type: {type(ix)}")

    def _freeze_all(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if "norm" not in name:
                param.requires_grad = False

    def _unfreeze_by_index(self, model: nn.Module, index: int) -> None:
        if not self.module_path:
            raise ValueError("module_path required for index-based freezing")

        module = model
        for part in self.module_path.split("."):
            module = getattr(module, part)

        if not isinstance(module, nn.ModuleList):
            raise TypeError(f"{self.module_path} must be a ModuleList")

        for param in module[index].parameters():
            param.requires_grad = True
