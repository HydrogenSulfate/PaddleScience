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


def _dataloader_cfg(batch_size=4, iters_per_epoch=1):
    return {
        "dataset": "IterableNamedArrayDataset",
        "iters_per_epoch": iters_per_epoch,
        "batch_size": batch_size,
    }


def _build(geom=None, **kwargs):
    if geom is None:
        geom = geometry.Rectangle((0, 0), (1, 1))
    return constraint.InteriorConstraint(
        {"u": lambda out: out["u"]},
        {"u": 0},
        geom,
        _dataloader_cfg(),
        loss.MSELoss("mean"),
        **kwargs,
    )


def test_interior_input_keys_and_shapes():
    np.random.seed(0)
    eq = _build()
    input_dict, label_dict, _ = next(iter(eq.data_iter))
    assert set(input_dict) == {"x", "y", "sdf"}
    for key in input_dict:
        assert tuple(input_dict[key].shape) == (4, 1)
    assert tuple(label_dict["u"].shape) == (4, 1)


def test_interior_default_weight_is_none():
    np.random.seed(0)
    eq = _build()
    _, _, weight_dict = next(iter(eq.data_iter))
    assert weight_dict is None


def test_interior_sdf_weight_matches_input_sdf():
    np.random.seed(0)
    eq = _build(weight_dict={"u": "sdf"})
    input_dict, _, weight_dict = next(iter(eq.data_iter))
    assert tuple(weight_dict["u"].shape) == tuple(input_dict["sdf"].shape)
    np.testing.assert_allclose(
        np.asarray(weight_dict["u"]), np.asarray(input_dict["sdf"])
    )


def test_interior_compute_sdf_derivatives_keys():
    np.random.seed(0)
    eq = _build(compute_sdf_derivatives=True)
    input_dict, _, _ = next(iter(eq.data_iter))
    assert "sdf__x" in input_dict
    assert "sdf__y" in input_dict
    assert "sdf__t" not in input_dict
    assert tuple(input_dict["sdf__x"].shape) == (4, 1)
    assert tuple(input_dict["sdf__y"].shape) == (4, 1)


def test_interior_missing_sdf_raises_value_error():
    # Hypercube does not implement sdf_func, so sample_interior returns no sdf.
    geom = geometry.Hypercube((0, 0), (1, 1))
    with pytest.raises(ValueError) as excinfo:
        _build(geom=geom, weight_dict={"u": "sdf"})
    message = str(excinfo.value)
    assert "Missing 'sdf' field in input" in message
    assert "geometry (Hypercube)" in message
    assert "sdf_func" in message


def test_interior_invalid_string_weight_raises():
    with pytest.raises(NotImplementedError) as excinfo:
        _build(weight_dict={"u": "not_a_field"})
    assert str(excinfo.value) == "string not_a_field is invalid yet."


if __name__ == "__main__":
    pytest.main()
