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

import numpy as np
import paddle
import pytest

import ppsci


def _supervised_cfg(with_weight=True, num_samples=8, batch_size=4):
    input_ = {"x": np.arange(num_samples, dtype="float32").reshape(-1, 1)}
    label = {"u": np.arange(num_samples, dtype="float32").reshape(-1, 1)}
    dataset_cfg = {
        "name": "NamedArrayDataset",
        "input": input_,
        "label": label,
    }
    if with_weight:
        dataset_cfg["weight"] = {
            "u": np.ones((num_samples, 1), dtype="float32"),
        }
    return {
        "dataset": dataset_cfg,
        "batch_size": batch_size,
        "num_workers": 0,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": False,
            "drop_last": False,
        },
    }


def _make_supervised_validator(with_weight=True, output_expr=None):
    return ppsci.validate.SupervisedValidator(
        _supervised_cfg(with_weight=with_weight),
        ppsci.loss.MSELoss("mean"),
        output_expr,
        metric={"MSE": ppsci.metric.MSE()},
        name="sup_validator",
    )


def test_supervised_validator_attributes_and_batch():
    validator = _make_supervised_validator()

    assert validator.input_keys == ("x",)
    assert validator.output_keys == ("u",)
    assert validator.name == "sup_validator"
    assert isinstance(validator.loss, ppsci.loss.MSELoss)
    assert list(validator.metric.keys()) == ["MSE"]
    assert len(validator.data_loader) == 2

    batch = next(iter(validator.data_loader))
    assert isinstance(batch, list)
    assert len(batch) == 3
    assert batch[0]["x"].shape == (4, 1)
    assert batch[1]["u"].shape == (4, 1)
    assert batch[2]["u"].shape == (4, 1)


def test_supervised_validator_batch_without_weight():
    validator = _make_supervised_validator(with_weight=False)

    batch = next(iter(validator.data_loader))
    assert batch[0]["x"].shape == (4, 1)
    assert batch[1]["u"].shape == (4, 1)
    assert batch[2] == {}


def test_supervised_validator_default_output_expr():
    validator = _make_supervised_validator(with_weight=False, output_expr=None)

    assert validator.output_keys == ("u",)
    output = {"u": paddle.to_tensor([[1.0], [2.0]], dtype="float32")}
    result = validator.output_expr["u"](output)
    assert paddle.equal_all(result, output["u"]).item()


def test_supervised_validator_str():
    validator = _make_supervised_validator()
    text = str(validator)

    assert "SupervisedValidator" in text
    assert "name = sup_validator" in text
    assert "input_keys = ('x',)" in text
    assert "len(dataloader) = 2" in text
    assert "metric = ['MSE']" in text


def test_supervised_validator_loss_and_metric():
    validator = _make_supervised_validator()
    output = {"u": paddle.zeros((2, 1), dtype="float32")}
    label = {"u": paddle.ones((2, 1), dtype="float32")}

    loss = validator.loss(output, label)
    metric = validator.metric["MSE"](output, label)

    assert loss["u"].item() == pytest.approx(1.0, rel=1e-6)
    assert metric["u"].item() == pytest.approx(1.0, rel=1e-6)


def _make_geometry_validator(label_dict, output_expr=None, total_size=8):
    if output_expr is None:
        output_expr = {"u": lambda out: out["u"]}
    return ppsci.validate.GeometryValidator(
        output_expr,
        label_dict,
        ppsci.geometry.Rectangle((0, 0), (1, 1)),
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "total_size": total_size,
            "batch_size": total_size,
        },
        ppsci.loss.MSELoss("mean"),
        metric={"MAE": ppsci.metric.MAE()},
        name="geo_validator",
    )


def test_geometry_validator_attributes_and_batch():
    validator = _make_geometry_validator({"u": 0.0})

    assert validator.input_keys == ("x", "y")
    assert validator.output_keys == ("u",)
    assert validator.name == "geo_validator"
    # Iterable dataset is returned directly by build_dataloader.
    assert len(validator.data_loader) == 1
    assert list(validator.metric.keys()) == ["MAE"]

    batch = next(iter(validator.data_loader))
    assert isinstance(batch, tuple)
    assert len(batch) == 3
    # Geometry sampling also provides an 'sdf' input beyond dim_keys.
    assert set(batch[0].keys()) == {"x", "y", "sdf"}
    for value in batch[0].values():
        assert value.shape == (8, 1)
    assert batch[1]["u"].shape == (8, 1)
    assert batch[2]["u"].shape == (8, 1)
    assert float(batch[2]["u"].min()) == pytest.approx(1.0)
    assert float(batch[2]["u"].max()) == pytest.approx(1.0)


def test_geometry_validator_callable_label():
    validator = _make_geometry_validator({"u": lambda inp: inp["x"] * 2})
    batch = next(iter(validator.data_loader))

    assert paddle.allclose(batch[1]["u"], batch[0]["x"] * 2).item()


def test_geometry_validator_scalar_label():
    validator = _make_geometry_validator({"u": 0.0})
    batch = next(iter(validator.data_loader))

    assert batch[1]["u"].shape == (8, 1)
    assert float(batch[1]["u"].abs().max()) == pytest.approx(0.0)


def test_geometry_validator_str():
    validator = _make_geometry_validator({"u": 0.0})
    text = str(validator)

    assert "GeometryValidator" in text
    assert "input_keys = ('x', 'y')" in text
    assert "len(dataloader) = 1" in text
    assert "metric = ['MAE']" in text


if __name__ == "__main__":
    pytest.main()
