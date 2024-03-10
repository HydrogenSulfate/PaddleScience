# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
from paddle import io
from paddle import vision

from ppsci import autodiff
from ppsci.utils import logger


class NamedArrayDataset(io.Dataset):
    """Class for Named Array Dataset.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Optional[Dict[str, np.ndarray]]): Label dict. Defaults to None.
        weight (Optional[Dict[str, np.ndarray]]): Weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> output = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.NamedArrayDataset(input, output, weight)
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Optional[Dict[str, np.ndarray]] = None,
        weight: Optional[Dict[str, np.ndarray]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = input
        self.label = {} if label is None else label
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(self.label.keys())
        self.weight = {} if weight is None else weight
        self.transforms = transforms
        self._len = len(next(iter(input.values())))

    def __getitem__(self, idx):
        input_item = {key: value[idx] for key, value in self.input.items()}
        label_item = {key: value[idx] for key, value in self.label.items()}
        weight_item = {key: value[idx] for key, value in self.weight.items()}

        if self.transforms is not None:
            input_item, label_item, weight_item = self.transforms(
                input_item, label_item, weight_item
            )

        return (input_item, label_item, weight_item)

    def __len__(self):
        return self._len


class IterableNamedArrayDataset(io.IterableDataset):
    """IterableNamedArrayDataset for full-data loading.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Optional[Dict[str, np.ndarray]]): Label dict. Defaults to None.
        weight (Optional[Dict[str, np.ndarray]]): Weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> label = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.IterableNamedArrayDataset(input, label, weight)
    """

    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = False

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Optional[Dict[str, np.ndarray]] = None,
        weight: Optional[Dict[str, np.ndarray]] = None,
        transforms: Optional[vision.Compose] = None,
    ):
        super().__init__()
        self.input = {key: paddle.to_tensor(value) for key, value in input.items()}
        self.label = (
            {key: paddle.to_tensor(value) for key, value in label.items()}
            if label is not None
            else {}
        )
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(self.label.keys())
        self.weight = (
            {
                key: paddle.to_tensor(value, paddle.get_default_dtype())
                for key, value in weight.items()
            }
            if weight is not None
            else None
        )
        self._len = len(next(iter(self.input.values())))
        self.transforms = transforms

    @property
    def num_samples(self):
        """Number of samples within current dataset."""
        return self._len

    def __iter__(self):
        if callable(self.transforms):
            input_, label_, weight_ = self.transforms(
                self.input, self.label, self.weight
            )
            yield input_, label_, weight_
        else:
            yield self.input, self.label, self.weight

    def __len__(self):
        return 1


class ImportanceSamplingIterableNamedArrayDataset(io.IterableDataset):
    """ImportanceSamplingIterableNamedArrayDataset for full-data loading.

    Args:
        input (Dict[str, np.ndarray]): Input dict.
        label (Optional[Dict[str, np.ndarray]]): Label dict. Defaults to None.
        batch_size (int): Batch size.
        weight (Optional[Dict[str, np.ndarray]]): Weight dict. Defaults to None.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.

    Examples:
        >>> import ppsci
        >>> input = {"x": np.random.randn(100, 1)}
        >>> label = {"u": np.random.randn(100, 1)}
        >>> weight = {"u": np.random.randn(100, 1)}
        >>> dataset = ppsci.data.dataset.IterableNamedArrayDataset(input, label, weight)
    """

    def __init__(
        self,
        input: Dict[str, np.ndarray],
        label: Optional[Dict[str, np.ndarray]],
        batch_size: int,
        importance_measure: Callable,
        weight: Optional[Dict[str, np.ndarray]] = None,
        transforms: Optional[vision.Compose] = None,
        resample_freq: int = 1000,
    ):
        super().__init__()
        self.input = input
        self.label = label
        self.input_keys = tuple(input.keys())
        self.label_keys = tuple(self.label.keys())
        self.weight = weight
        self._len = len(next(iter(self.input.values())))
        self.transforms = transforms
        self.resample_freq = resample_freq
        self.batch_size = min(batch_size, self.num_samples)
        self.importance_measure = importance_measure

        def iterable_function():
            counter = 0
            while True:
                # resample all points when needed
                if counter % self.resample_freq == 0:
                    logger.message(
                        f"Staring importance sampling(resample_freq={self.resample_freq}), "
                        "this may take a while..."
                    )
                    importance = []
                    batch_list_input: Dict[str, List[np.ndarray]] = {
                        k: np.split(v, v.shape[0] // self.batch_size)
                        for k, v in self.input.items()
                    }

                    for i in range(len(next(iter(batch_list_input.values())))):
                        batch_importance = self.importance_measure(
                            {
                                k: paddle.to_tensor(v[i], stop_gradient=False)
                                for k, v in batch_list_input.items()
                            }
                        ).numpy()
                        importance.append(batch_importance)
                        # NOTE: Clear grad cache after every batch, or memory will be exhausted soon.
                        autodiff.clear()
                    importance = np.concatenate(importance, axis=0)
                    prob = importance / np.sum(self.input["area"] * importance)

                # sample points from probability distribution and store idx
                idx = np.array([])
                while True:
                    r = np.random.uniform(0, np.max(prob), size=self.batch_size)
                    try_idx = np.random.choice(self.num_samples, self.batch_size)
                    if_sample = np.less(r, prob[try_idx, :][:, 0])
                    idx = np.concatenate([idx, try_idx[if_sample]])
                    if idx.shape[0] >= batch_size:
                        idx = idx[:batch_size]
                        break
                idx = idx.astype(np.int64)

                # gather input, label, and weight
                input = {k: v[idx] for k, v in self.input.items()}
                label = {k: v[idx] for k, v in self.label.items()}
                weight = (
                    {k: v[idx] for k, v in self.weight.items()}
                    if self.weight is not None
                    else None
                )

                # set area value from importance sampling
                input["area"] = 1.0 / (prob[idx] * batch_size)
                # return and count up
                counter += 1

                input = {k: paddle.to_tensor(v) for k, v in input.items()}
                label = {k: paddle.to_tensor(v) for k, v in label.items()}
                weight = (
                    {k: paddle.to_tensor(v) for k, v in weight.items()}
                    if weight is not None
                    else None
                )
                yield (input, label, weight)

        self.iterable_function = iterable_function

    @property
    def num_samples(self):
        """Number of samples within current dataset."""
        return self._len

    def __len__(self):
        return 100000000

    def __iter__(self):
        yield from self.iterable_function()
