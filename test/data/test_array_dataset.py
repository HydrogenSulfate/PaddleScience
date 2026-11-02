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

from ppsci.data import build_dataloader
from ppsci.data import register_to_dataset
from ppsci.data.dataset import IterableNamedArrayDataset
from ppsci.data.dataset import NamedArrayDataset
from ppsci.data.dataset import build_dataset
from ppsci.data.process.transform import Compose


def _make_arrays(num_samples=10):
    input_ = {
        "x": np.arange(num_samples, dtype="float32").reshape(-1, 1),
        "y": np.arange(num_samples, dtype="float32").reshape(-1, 1) * 2,
    }
    label = {"u": np.arange(num_samples, dtype="float32").reshape(-1, 1)}
    weight = {"u": np.ones((num_samples, 1), dtype="float32")}
    return input_, label, weight


def test_named_array_dataset_len_and_keys():
    input_, label, _ = _make_arrays()
    dataset = NamedArrayDataset(input_, label)

    assert len(dataset) == 10
    assert dataset.input_keys == ("x", "y")
    assert dataset.label_keys == ("u",)
    assert NamedArrayDataset.batch_index is True


def test_named_array_dataset_getitem_int():
    input_, label, weight = _make_arrays()
    dataset = NamedArrayDataset(input_, label, weight)

    input_item, label_item, weight_item = dataset[2]

    assert isinstance(input_item, dict)
    assert isinstance(label_item, dict)
    assert isinstance(weight_item, dict)
    np.testing.assert_array_equal(input_item["x"], np.array([2.0], dtype="float32"))
    np.testing.assert_array_equal(input_item["y"], np.array([4.0], dtype="float32"))
    np.testing.assert_array_equal(label_item["u"], np.array([2.0], dtype="float32"))
    np.testing.assert_array_equal(weight_item["u"], np.array([1.0], dtype="float32"))


@pytest.mark.parametrize("indices", [[0, 2, 4], np.array([0, 2, 4])])
def test_named_array_dataset_getitem_batch_indices(indices):
    input_, label, weight = _make_arrays()
    dataset = NamedArrayDataset(input_, label, weight)

    input_item, label_item, weight_item = dataset[indices]

    assert input_item["x"].shape == (3, 1)
    assert input_item["y"].shape == (3, 1)
    assert label_item["u"].shape == (3, 1)
    assert weight_item["u"].shape == (3, 1)
    np.testing.assert_array_equal(
        input_item["x"].ravel(), np.array([0.0, 2.0, 4.0], dtype="float32")
    )


def test_named_array_dataset_without_label_and_weight():
    input_, _, _ = _make_arrays()
    dataset = NamedArrayDataset(input_)

    assert dataset.label_keys == ()
    assert dataset.label == {}
    assert dataset.weight == {}
    assert len(dataset) == 10

    input_item, label_item, weight_item = dataset[0]
    assert set(input_item.keys()) == {"x", "y"}
    assert label_item == {}
    assert weight_item == {}


def test_named_array_dataset_with_custom_transform():
    input_, label, _ = _make_arrays()

    def double_transform(input_dict, label_dict, weight_dict):
        return (
            {key: value * 2 for key, value in input_dict.items()},
            {key: value * 2 for key, value in label_dict.items()},
            weight_dict,
        )

    dataset = NamedArrayDataset(
        input_,
        label,
        transforms=Compose([double_transform]),
    )

    input_item, label_item, _ = dataset[0]
    np.testing.assert_array_equal(input_item["x"], np.array([0.0], dtype="float32"))
    np.testing.assert_array_equal(label_item["u"], np.array([0.0], dtype="float32"))

    input_item, label_item, _ = dataset[1]
    np.testing.assert_array_equal(input_item["x"], np.array([2.0], dtype="float32"))
    np.testing.assert_array_equal(label_item["u"], np.array([2.0], dtype="float32"))

    # The transform must not mutate the source arrays.
    np.testing.assert_array_equal(input_["x"].ravel(), np.arange(10, dtype="float32"))
    np.testing.assert_array_equal(label["u"].ravel(), np.arange(10, dtype="float32"))


def test_iterable_named_array_dataset_yields_full_batch():
    input_, label, _ = _make_arrays()
    dataset = IterableNamedArrayDataset(input_, label)

    assert len(dataset) == 1
    assert dataset.num_samples == 10
    assert IterableNamedArrayDataset.batch_index is False

    input_item, label_item, weight_item = next(iter(dataset))
    assert isinstance(input_item, dict)
    assert isinstance(label_item, dict)
    assert weight_item is None
    for value in input_item.values():
        assert isinstance(value, paddle.Tensor)
        assert value.shape == (10, 1)
    assert label_item["u"].shape == (10, 1)


def test_iterable_named_array_dataset_transform_and_dataloader_passthrough():
    input_, label, _ = _make_arrays()

    def double_transform(input_dict, label_dict, weight_dict):
        return (
            {key: value * 2 for key, value in input_dict.items()},
            {key: value * 2 for key, value in label_dict.items()},
            weight_dict,
        )

    dataset = IterableNamedArrayDataset(
        input_, label, transforms=Compose([double_transform])
    )
    input_item, label_item, _ = next(iter(dataset))
    np.testing.assert_array_equal(
        input_item["x"].numpy().ravel(), np.arange(10, dtype="float32") * 2
    )
    np.testing.assert_array_equal(
        label_item["u"].numpy().ravel(), np.arange(10, dtype="float32") * 2
    )

    # build_dataloader returns iterable datasets unchanged.
    dataloader = build_dataloader(dataset, {"batch_size": 4})
    assert dataloader is dataset


def test_build_dataloader_with_batch_sampler():
    input_, label, _ = _make_arrays()
    dataset = NamedArrayDataset(input_, label)
    cfg = {
        # No "dataset" key is needed here; batch_index datasets used to raise
        # KeyError when auto_collation was not given explicitly.
        "batch_size": 4,
        "num_workers": 0,
        "sampler": {
            "name": "BatchSampler",
            "shuffle": False,
            "drop_last": False,
        },
    }

    dataloader = build_dataloader(dataset, cfg)
    assert len(dataloader) == 3

    batches = list(dataloader)
    assert len(batches) == 3
    # Auto-collation is disabled, so each batch is a Python list of 3 dicts.
    assert isinstance(batches[0], list)
    for batch in batches:
        assert isinstance(batch, list)
        assert len(batch) == 3

    first = batches[0]
    assert first[0]["x"].shape == (4, 1)
    assert first[1]["u"].shape == (4, 1)
    assert first[2] == {}
    np.testing.assert_array_equal(
        first[0]["x"].numpy().ravel(), np.array([0.0, 1.0, 2.0, 3.0], dtype="float32")
    )
    assert batches[-1][0]["x"].shape == (2, 1)


def test_build_dataset_named_array_dataset():
    input_, label, weight = _make_arrays()
    cfg = {
        "name": "NamedArrayDataset",
        "input": input_,
        "label": label,
        "weight": weight,
    }

    dataset = build_dataset(cfg)
    assert isinstance(dataset, NamedArrayDataset)
    assert len(dataset) == 10
    assert dataset.input_keys == ("x", "y")
    assert dataset.label_keys == ("u",)

    # An existing dataset instance is returned unchanged.
    assert build_dataset(dataset) is dataset


def test_build_dataset_registered_custom_class():
    class TinyDatasetForTest(paddle.io.Dataset):
        def __init__(self, num_samples=4):
            super().__init__()
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, index):
            return index

    # NOTE: register_to_dataset returns None, so register explicitly instead of
    # using it as a decorator.
    register_to_dataset(TinyDatasetForTest)

    dataset = build_dataset({"name": "TinyDatasetForTest", "num_samples": 5})
    assert isinstance(dataset, TinyDatasetForTest)
    assert len(dataset) == 5


if __name__ == "__main__":
    pytest.main()
