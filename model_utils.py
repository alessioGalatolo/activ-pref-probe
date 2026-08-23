"""Model utility functions for APO."""

from typing import List
import torch.nn as nn


def get_model_layers(model) -> List[nn.Module]:
    """Get the transformer layers of the model."""
    if hasattr(model, 'layers'):
        return model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    else:
        raise ValueError("Cannot find model layers.")
