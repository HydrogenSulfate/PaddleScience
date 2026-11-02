# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import pytest

from ppsci import constraint
from ppsci import geometry
from ppsci import loss


def _dataloader_cfg(batch_size=4, iters_per_epoch=2, integral_batch_size=3):
    return {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
        "batch_size": batch_size,
        "integral_batch_size": integral_batch_size,
    }


def _build(geom=None, label_dict=None, **kwargs):
    if geom is None:
        geom = geometry.Rectangle((0, 0), (1, 1))
    if label_dict is None:
        label_dict = {"u": 0}
    return constraint.IntegralConstraint(
        {"u": lambda out: out["u"]},
        label_dict,
        geom,
        _dataloader_cfg(),
        loss.MSELoss("mean"),
        **kwargs,
    )


def test_integral_input_and_int_label_shapes():
    np.random.seed(0)
    igc = _build()
    input_dict, label_dict, _ = next(iter(igc.data_iter))

    # batch_size * iters_per_epoch = 8 samples, each with integral_batch_size=3
    for key in ("x", "y", "normal_x", "normal_y"):
        assert key in input_dict
        assert tuple(input_dict[key].shape) == (8, 3, 1)
    assert tuple(label_dict["u"].shape) == (8, 1)
    np.testing.assert_allclose(np.asarray(label_dict["u"]), 0.0)


def test_integral_callable_array_label_shape():
    np.random.seed(0)
    igc = _build(label_dict={"u": lambda d: d["x"]})
    input_dict, label_dict, _ = next(iter(igc.data_iter))
    assert tuple(label_dict["u"].shape) == (8, 3, 1)
    np.testing.assert_allclose(np.asarray(label_dict["u"]), np.asarray(input_dict["x"]))


def test_integral_callable_scalar_label_shape():
    np.random.seed(0)
    igc = _build(label_dict={"u": lambda d: 1.5})
    _, label_dict, _ = next(iter(igc.data_iter))
    assert tuple(label_dict["u"].shape) == (8, 1)
    np.testing.assert_allclose(np.asarray(label_dict["u"]), 1.5)


def test_integral_no_area_scaling():
    np.random.seed(0)

    class _AreaGeom:
        dim_keys = ("x", "y")

        def sample_boundary(self, n, random, criteria):
            return {
                "x": np.zeros((n, 1), dtype=paddle.get_default_dtype()),
                "y": np.zeros((n, 1), dtype=paddle.get_default_dtype()),
                "area": np.full((n, 1), 5.0, dtype=paddle.get_default_dtype()),
            }

    igc = _build(geom=_AreaGeom())
    input_dict, label_dict, _ = next(iter(igc.data_iter))

    # 8 stacked samples of (3, 1); prepare_integral_input does not scale area.
    assert tuple(input_dict["area"].shape) == (8, 3, 1)
    np.testing.assert_allclose(np.asarray(input_dict["area"]), 5.0)
    assert tuple(label_dict["u"].shape) == (8, 1)


if __name__ == "__main__":
    pytest.main()
