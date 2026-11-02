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

import math

import numpy as np
import paddle
import pytest

from ppsci.metric import MAE
from ppsci.metric import MSE
from ppsci.metric import RMSE
from ppsci.metric import FunctionalMetric
from ppsci.metric import L2Rel
from ppsci.metric import MaxAE
from ppsci.metric import MeanL2Rel
from ppsci.metric import R2Score
from ppsci.metric import build_metric
from ppsci.utils.misc import PrettyOrderedDict


@pytest.fixture
def output_label():
    output = {"u": paddle.to_tensor([[1.0, 2.0], [3.0, 5.0]], dtype="float32")}
    label = {"u": paddle.to_tensor([[2.0, 2.0], [3.0, 3.0]], dtype="float32")}
    return output, label


def test_mse(output_label):
    output, label = output_label

    result = MSE()(output, label)
    assert result["u"].item() == pytest.approx(1.25, rel=1e-6)

    result = MSE(keep_batch=True)(output, label)
    assert result["u"].shape == (2,)
    assert result["u"][0].item() == pytest.approx(0.5, rel=1e-6)
    assert result["u"][1].item() == pytest.approx(2.0, rel=1e-6)


def test_mae(output_label):
    output, label = output_label

    result = MAE()(output, label)
    assert result["u"].item() == pytest.approx(0.75, rel=1e-6)

    result = MAE(keep_batch=True)(output, label)
    assert result["u"].shape == (2,)
    assert result["u"][0].item() == pytest.approx(0.5, rel=1e-6)
    assert result["u"][1].item() == pytest.approx(1.0, rel=1e-6)


def test_rmse(output_label):
    output, label = output_label

    result = RMSE()(output, label)
    assert result["u"].item() == pytest.approx(math.sqrt(1.25), rel=1e-6)

    with pytest.raises(ValueError, match="keep_batch should be False"):
        RMSE(keep_batch=True)


def test_l2_rel(output_label):
    output, label = output_label

    result = L2Rel()(output, label)
    expected = math.sqrt(5) / math.sqrt(26)
    assert result["u"].item() == pytest.approx(expected, rel=1e-5)

    with pytest.raises(ValueError, match="keep_batch should be False"):
        L2Rel(keep_batch=True)


def test_mean_l2_rel(output_label):
    output, label = output_label

    result = MeanL2Rel()(output, label)
    assert result["u"].item() == pytest.approx(0.412479, rel=1e-5)

    result = MeanL2Rel(keep_batch=True)(output, label)
    assert result["u"].shape == (2,)
    assert result["u"][0].item() == pytest.approx(0.353553, rel=1e-5)
    assert result["u"][1].item() == pytest.approx(0.471405, rel=1e-5)


def test_r2_score(output_label):
    output, label = output_label

    result = R2Score()(output, label)
    assert result["u"].item() == pytest.approx(-4.0, rel=1e-6)

    perfect = R2Score()(output, output)
    assert perfect["u"].item() == pytest.approx(1.0, abs=1e-6)


def test_r2_score_keep_batch_shape(output_label):
    output, label = output_label

    result = R2Score(keep_batch=True)(output, label)
    # Constant targets still use the original denominator stabilization.
    assert result["u"].shape == (2,)
    np.testing.assert_allclose(result["u"].numpy(), [-1e8, -4e8], rtol=1e-6)


def test_r2_score_keep_batch_values():
    output = {"u": paddle.to_tensor([[1.0, 2.0], [3.0, 5.0]])}
    label = {"u": paddle.to_tensor([[1.0, 3.0], [2.0, 4.0]])}

    result = R2Score(keep_batch=True)(output, label)
    # row 0: 1 - 1/2 = 0.5; row 1: 1 - 2/2 = 0.0
    np.testing.assert_allclose(result["u"].numpy(), [0.5, 0.0], atol=1e-6)


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
def test_r2_score_keep_batch_scalar_samples(shape):
    output = {"u": paddle.to_tensor([1.0, 2.0]).reshape(shape)}
    label = {"u": paddle.to_tensor([1.0, 3.0]).reshape(shape)}

    result = R2Score(keep_batch=True)(output, label)
    assert result["u"].shape == (2,)
    np.testing.assert_allclose(result["u"].numpy(), [1.0, -1e8], rtol=1e-6)


def test_r2_score_keep_batch_multiple_feature_axes():
    output = {
        "u": paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    }
    label = {
        "u": paddle.to_tensor([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]])
    }

    result = R2Score(keep_batch=True)(output, label)

    assert result["u"].shape == (2,)
    np.testing.assert_allclose(result["u"].numpy(), [0.6, 0.6], rtol=1e-6)


@pytest.mark.parametrize(
    "target,prediction,expected",
    [
        ([2.0] * 4, [3.0] * 4, -399999999.0),
        ([2.0, 2.0], [2.0, 2.0], 1.0),
        ([2.0], [3.0], -99999999.0),
        ([0.0, 1e-7], [0.0, 1.0], -99999929.000036),
        ([2.0, 2.0], [2.0 + 1e-7, 2.0], 0.9999990000000033),
    ],
)
def test_r2_score_preserves_original_numerical_behavior(target, prediction, expected):
    output = {"u": paddle.to_tensor(prediction, dtype="float64").reshape([-1, 1])}
    label = {"u": paddle.to_tensor(target, dtype="float64").reshape([-1, 1])}

    result = R2Score()(output, label)

    assert result["u"].shape == ()
    assert result["u"].item() == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_max_ae(output_label):
    output, label = output_label

    result = MaxAE()(output, label)
    assert result["u"].item() == pytest.approx(2.0, rel=1e-6)

    result = MaxAE(keep_batch=True)(output, label)
    assert result["u"].shape == (2,)
    assert result["u"][0].item() == pytest.approx(1.0, rel=1e-6)
    assert result["u"][1].item() == pytest.approx(2.0, rel=1e-6)


def test_build_metric():
    metrics = build_metric([{"MSE": {}}, {"MAE": {}}])

    assert isinstance(metrics, PrettyOrderedDict)
    assert list(metrics.keys()) == ["MSE", "MAE"]
    assert isinstance(metrics["MSE"], MSE)
    assert isinstance(metrics["MAE"], MAE)


def test_functional_metric(output_label):
    output, label = output_label

    metric = FunctionalMetric(
        lambda output_dict, label_dict: {
            "mae": (output_dict["u"] - label_dict["u"]).abs().mean()
        }
    )
    result = metric(output, label)
    assert result["mae"].item() == pytest.approx(0.75, rel=1e-6)

    # A 0-dim result is rejected when keep_batch is True.
    scalar_metric = FunctionalMetric(
        lambda output_dict, label_dict: (output_dict["u"] - label_dict["u"])
        .abs()
        .mean(),
        keep_batch=True,
    )
    with pytest.raises(AssertionError):
        scalar_metric(output, label)

    # A per-sample result is accepted and kept as-is.
    per_sample_metric = FunctionalMetric(
        lambda output_dict, label_dict: (
            (output_dict["u"] - label_dict["u"]) ** 2
        ).mean(axis=1),
        keep_batch=True,
    )
    result = per_sample_metric(output, label)
    assert result.shape == (2,)
    assert result[0].item() == pytest.approx(0.5, rel=1e-6)
    assert result[1].item() == pytest.approx(2.0, rel=1e-6)


if __name__ == "__main__":
    pytest.main()
