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
import pytest

from ppsci import constraint
from ppsci import geometry
from ppsci import loss


def _time_geom():
    return geometry.TimeXGeometry(
        geometry.TimeDomain(0, 1), geometry.Rectangle((0, 0), (1, 1))
    )


def _build(**kwargs):
    return constraint.InitialConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        _time_geom(),
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "batch_size": 4,
        },
        loss.MSELoss("mean"),
        **kwargs,
    )


def test_initial_input_keys_and_shapes():
    np.random.seed(0)
    ic = _build()
    assert ic.input_keys == ("t", "x", "y")
    assert ic.output_keys == ("u",)

    input_dict, label_dict, _ = next(iter(ic.data_iter))
    for key in ("t", "x", "y", "sdf"):
        assert key in input_dict
        assert tuple(input_dict[key].shape) == (4, 1)
    assert tuple(label_dict["u"].shape) == (4, 1)


def test_initial_time_is_initial_time():
    np.random.seed(0)
    ic = _build()
    input_dict, _, _ = next(iter(ic.data_iter))
    np.testing.assert_allclose(np.asarray(input_dict["t"]), 0.0)


def test_initial_sdf_weight_matches_input_sdf():
    np.random.seed(0)
    ic = _build(weight_dict={"u": "sdf"})
    input_dict, _, weight_dict = next(iter(ic.data_iter))
    assert tuple(weight_dict["u"].shape) == tuple(input_dict["sdf"].shape)
    np.testing.assert_allclose(
        np.asarray(weight_dict["u"]), np.asarray(input_dict["sdf"])
    )


def test_initial_compute_sdf_derivatives_spatial_only():
    np.random.seed(0)
    ic = _build(compute_sdf_derivatives=True)
    input_dict, _, _ = next(iter(ic.data_iter))
    assert "sdf__x" in input_dict
    assert "sdf__y" in input_dict
    assert "sdf__t" not in input_dict
    assert tuple(input_dict["sdf__x"].shape) == (4, 1)
    assert tuple(input_dict["sdf__y"].shape) == (4, 1)


if __name__ == "__main__":
    pytest.main()
