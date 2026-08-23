"""Activation extraction utilities for APO."""

from typing import List
import torch
from model_utils import get_model_layers


class ActivationExtractor:
    """Extract activations from specified layers during forward pass."""

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.layer_indices:
            layer = get_model_layers(self.model)[idx]
            hook = layer.register_forward_hook(self._get_hook(idx))
            self.hooks.append(hook)

    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            # output[0] is hidden states: [batch, seq_len, hidden_dim]
            # Take the last token's activation (for causal LM)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self.activations[layer_idx] = hidden[:, -1, :].detach().cpu()
        return hook

    def extract(self, input_ids):
        self.activations = {}
        with torch.no_grad():
            self.model(input_ids=input_ids)
        # Concatenate activations from all layers
        acts = [self.activations[idx] for idx in sorted(self.activations.keys())]
        return torch.cat(acts, dim=-1)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
