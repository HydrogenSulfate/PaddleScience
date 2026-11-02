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

"""Numeric tests for MSE/MAE/L1/L2 family of losses."""

import numpy as np
import paddle
import pytest

from ppsci.loss import CausalMSELoss
from ppsci.loss import L1Loss
from ppsci.loss import L2Loss
from ppsci.loss import MAELoss
from ppsci.loss import MSELoss
from ppsci.loss import MSELossWithL2Decay
from ppsci.loss import PeriodicMSELoss


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


def _assert_losses(result, expected, rtol=1e-5, atol=1e-5):
    assert set(result) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(
            float(result[key]), value, rtol=rtol, atol=atol, err_msg=key
        )


def test_mse_mean_exact_value():
    output_dict, label_dict = _make_data()
    _assert_losses(MSELoss()(output_dict, label_dict), {"u": 5.3575, "v": 0.94})


def test_mse_sum_exact_value():
    output_dict, label_dict = _make_data()
    _assert_losses(
        MSELoss(reduction="sum")(output_dict, label_dict),
        {"u": 21.43, "v": 3.76},
    )


def test_mse_weight_dict_scalar_and_per_key():
    output_dict, label_dict = _make_data()
    weight = {"u": 0.8, "v": 0.2}
    _assert_losses(
        MSELoss(weight=weight)(output_dict, label_dict),
        {"u": 4.286, "v": 0.188},
    )
    _assert_losses(
        MSELoss(reduction="sum", weight=weight)(output_dict, label_dict),
        {"u": 17.144, "v": 0.752},
    )


def test_mse_weight_dict_argument_elementwise():
    output_dict, label_dict = _make_data()
    # weight_dict scales element-wise before reduction; use reduction='sum'
    # so that the expected value is 2 * sum_u = 42.86.
    result = MSELoss(reduction="sum")(output_dict, label_dict, weight_dict={"u": 2.0})
    _assert_losses(result, {"u": 42.86, "v": 3.76})


def test_mse_area_multiplication():
    output_dict, _ = _make_data()
    output_dict = {
        "u": output_dict["u"],
        "area": paddle.to_tensor([[2.0], [3.0]]),
    }
    labels = {"u": paddle.zeros([2, 2])}

    # sum_u(output^2) = 0.25 + 0.81 + 1.21 + 1.69 = 3.96, then scaled by
    # row area ([2], [3]): 2 * 1.06 + 3 * 2.9 = 10.82.
    result_sum = MSELoss(reduction="sum")(output_dict, labels)
    result_mean = MSELoss()(output_dict, labels)
    np.testing.assert_allclose(float(result_sum["u"]), 10.82, rtol=1e-5)
    np.testing.assert_allclose(float(result_mean["u"]), 2.705, rtol=1e-5)


def test_mse_label_keys_select_outputs():
    output_dict, label_dict = _make_data()
    result = MSELoss()(output_dict, {"u": label_dict["u"]})
    _assert_losses(result, {"u": 5.3575})


def test_mse_invalid_reduction_raises():
    with pytest.raises(ValueError):
        MSELoss(reduction="bad")
    with pytest.raises(ValueError):
        CausalMSELoss(n_chunks=0)


def test_mse_with_l2_decay_regularization():
    output_dict, label_dict = _make_data()
    output_dict = {
        **output_dict,
        "w": paddle.to_tensor([[0.5, 0.9], [1.1, -1.3]]),
    }

    result = MSELossWithL2Decay(regularization_dict={"w": 2.0})(output_dict, label_dict)
    _assert_losses(result, {"u": 5.3575, "v": 0.94, "w": 7.92})

    result = MSELossWithL2Decay(
        regularization_dict={"w": 2.0}, weight={"u": 0.8, "v": 0.2}
    )(output_dict, label_dict)
    _assert_losses(result, {"u": 4.286, "v": 0.188, "w": 7.92})


def test_mse_with_l2_decay_rejects_overlapping_key():
    output_dict, label_dict = _make_data()

    with pytest.raises(ValueError, match="must not overlap"):
        MSELossWithL2Decay(regularization_dict={"u": 2.0})(output_dict, label_dict)


def test_periodic_mse_loss_smoke():
    output_dict = {
        "u": paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
    }
    label_dict = {"u": paddle.zeros([4, 2])}
    np.testing.assert_allclose(
        float(PeriodicMSELoss()(output_dict, label_dict)["u"]), 0.0, atol=1e-6
    )

    odd_output = {"u": paddle.zeros([3, 2])}
    odd_label = {"u": paddle.zeros([3, 2])}
    with pytest.raises(ValueError):
        PeriodicMSELoss()(odd_output, odd_label)


def test_mae_loss_exact_values():
    output_dict, label_dict = _make_data()
    _assert_losses(MAELoss()(output_dict, label_dict), {"u": 1.875, "v": 0.9})
    _assert_losses(
        MAELoss(weight={"u": 0.8})(output_dict, label_dict),
        {"u": 1.5, "v": 0.9},
    )
    _assert_losses(
        MAELoss(reduction="sum", weight={"u": 0.8, "v": 0.2})(output_dict, label_dict),
        {"u": 6.0, "v": 0.72},
    )


def test_l1_loss_exact_values():
    output_dict, label_dict = _make_data()
    _assert_losses(
        L1Loss(weight={"u": 0.8})(output_dict, label_dict),
        {"u": 3.0, "v": 1.8},
    )
    _assert_losses(
        L1Loss(reduction="sum", weight={"u": 0.8})(output_dict, label_dict),
        {"u": 6.0, "v": 3.6},
    )


def test_l2_loss_exact_values():
    output_dict, label_dict = _make_data()
    _assert_losses(
        L2Loss(weight={"u": 0.8})(output_dict, label_dict),
        {"u": 2.5273559, "v": 1.3074461},
    )
    _assert_losses(
        L2Loss(reduction="sum", weight={"u": 0.8})(output_dict, label_dict),
        {"u": 5.0547118, "v": 2.6148922},
    )


@pytest.mark.parametrize("loss_cls", [MSELoss, MAELoss, L1Loss, L2Loss])
def test_invalid_reduction_all_simple_losses(loss_cls):
    with pytest.raises(ValueError):
        loss_cls(reduction="bad")


if __name__ == "__main__":
    pytest.main()
