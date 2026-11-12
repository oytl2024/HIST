"""Deterministic regression tests; run: python -m unittest discover -s tests -v.

The fixed encoder isolates aggregation from GRU training. Only device lookup is
adapted for CPU; the production aggregation and cosine operators run unchanged.
"""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from model import HIST


class FixedEncoder(nn.Module):
    def __init__(self, embeddings):
        """Input: [N,d] tensor; output: encoder fixture retaining its autograd graph."""
        super().__init__()
        self.embeddings = embeddings

    def forward(self, x):
        """Input: ignored GRU input; output: ([N,1,d], None), matching GRU's interface."""
        return self.embeddings.unsqueeze(1), None


def aggregation_model(embeddings, k=1):
    """Input: controlled [N,2] stock representations; output: HIST with zero backcasts."""
    model = HIST(d_feat=2, hidden_size=2, num_layers=1, K=k)
    model.rnn = FixedEncoder(embeddings)
    with torch.no_grad():
        for layer in (model.fc_ps_back, model.fc_hs_back):
            layer.weight.zero_()
            layer.bias.zero_()
    return model


def aggregate_input(model, layer, membership):
    """Input: model/layer/membership; output: actual input to selected aggregation head."""
    captured = []
    def capture(module, inputs):
        captured.append(inputs[0].detach().clone())
    handle = layer.register_forward_pre_hook(capture)
    try:
        n = membership.shape[0]
        with patch("torch.get_device", return_value="cpu"):
            model(torch.ones(n, 2), membership, torch.ones(n))
    finally:
        handle.remove()
    return captured[0]


class ConceptFilteringTests(unittest.TestCase):
    def test_predefined_nonzero_vector_with_zero_component_sum_is_kept(self):
        model = aggregation_model(torch.tensor([[0.8, -0.8]]))
        actual = aggregate_input(model, model.fc_ps, torch.ones(1, 1))
        torch.testing.assert_close(actual, torch.tensor([[0.8, -0.8]]))

    def test_hidden_nonzero_vectors_with_zero_component_sum_are_kept(self):
        model = aggregation_model(torch.tensor([[1., -1.], [1., -1.]]))
        actual = aggregate_input(model, model.fc_hs, torch.ones(2, 1))
        # Each hidden concept is the sum of two identical residual vectors.
        torch.testing.assert_close(actual, torch.tensor([[2., -2.], [2., -2.]]))

    def test_empty_predefined_column_is_still_filtered(self):
        model = aggregation_model(torch.tensor([[1., 0.]]))
        actual = aggregate_input(model, model.fc_ps, torch.tensor([[1., 0.]]))
        torch.testing.assert_close(actual, torch.tensor([[1., 0.]]))


if __name__ == "__main__":
    unittest.main()
