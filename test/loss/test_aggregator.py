# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Sum MTL aggregator and loss/aggregator builders."""

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from ppsci.loss import L1Loss
from ppsci.loss import L2Loss
from ppsci.loss import MAELoss
from ppsci.loss import MSELoss
from ppsci.loss import build_loss
from ppsci.loss import mtl


@pytest.fixture(autouse=True)
def _cpu_device():
    old_device = paddle.get_device()
    paddle.set_device("cpu")
    yield
    paddle.set_device(old_device)


def _make_data():
    output_dict = {
        "u": paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
        "v": paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
    }
    label_dict = {
        "u": paddle.to_tensor([[-1.8, 1.0], [-0.2, 2.5]]),
        "v": paddle.to_tensor([[0.1, 0.1], [0.1, 0.1]]),
    }
    return output_dict, label_dict


def test_sum_returns_exact_sum():
    aggregator = mtl.Sum()
    losses = {
        "a": paddle.to_tensor(1.0),
        "b": paddle.to_tensor(2.0),
        "c": paddle.to_tensor(3.0),
    }
    np.testing.assert_allclose(float(aggregator(losses)), 6.0, atol=1e-6)


def test_sum_single_loss():
    aggregator = mtl.Sum()
    loss = paddle.to_tensor(2.5)
    result = aggregator({"a": loss}, step=3)
    np.testing.assert_allclose(float(result), 2.5, atol=1e-6)
    assert aggregator.step == 3


def test_sum_step_stored():
    aggregator = mtl.Sum()
    aggregator({"a": paddle.to_tensor(1.0)}, step=7)
    assert aggregator.step == 7


def test_sum_empty_losses_raises():
    aggregator = mtl.Sum()
    with pytest.raises(AssertionError, match="can not be empty"):
        aggregator({})


def test_sum_gradient_flows():
    aggregator = mtl.Sum()
    a = paddle.to_tensor(1.0, stop_gradient=False)
    b = paddle.to_tensor(2.0, stop_gradient=False)
    result = aggregator({"a": a, "b": b})
    result.backward()

    np.testing.assert_allclose(a.grad.numpy(), 1.0)
    np.testing.assert_allclose(b.grad.numpy(), 1.0)


def test_sum_should_persist():
    assert mtl.Sum.should_persist is False


def test_sum_is_proper_nn_layer():
    aggregator = mtl.Sum()

    # Sum should still be a valid nn.Layer even though it does not need a model.
    assert aggregator.model is None
    assert "Sum" in repr(aggregator)
    assert list(aggregator.parameters()) == []
    assert dict(aggregator.state_dict()) == {}


def test_build_mtl_aggregator_from_config():
    aggregator = mtl.build_mtl_aggregator(OmegaConf.create({"name": "Sum"}))
    assert isinstance(aggregator, mtl.Sum)
    result = aggregator({"a": paddle.to_tensor(1.0), "b": paddle.to_tensor(2.0)})
    np.testing.assert_allclose(float(result), 3.0, atol=1e-6)


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        ("MSELoss", MSELoss),
        ("MAELoss", MAELoss),
        ("L1Loss", L1Loss),
        ("L2Loss", L2Loss),
    ],
)
def test_build_loss_from_config_names(name, expected_cls):
    loss = build_loss(OmegaConf.create({"name": name}))
    assert isinstance(loss, expected_cls)


def test_build_loss_with_kwargs():
    loss = build_loss(
        OmegaConf.create({"name": "MSELoss", "reduction": "sum", "weight": 0.5})
    )
    assert isinstance(loss, MSELoss)
    assert loss.reduction == "sum"
    assert loss.weight == 0.5

    output_dict, label_dict = _make_data()
    result = loss(output_dict, label_dict)
    np.testing.assert_allclose(float(result["u"]), 10.715, rtol=1e-5)


def test_build_loss_invalid_name_raises():
    with pytest.raises(NameError):
        build_loss(OmegaConf.create({"name": "NoSuchLoss"}))


if __name__ == "__main__":
    pytest.main()
