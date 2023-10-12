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

from typing import Optional

import numpy as np
from paddle import io
from paddle.io import BatchSampler
from paddle.io import DistributedBatchSampler

from ppsci.utils import misc

__all__ = [
    "BatchSampler",
    "DistributedBatchSampler",
    "FPSBatchSampler",
    "DistributedFPSBatchSampler",
]


class FPSBatchSampler(io.BatchSampler):
    """Batch sampler using Farthest Point Sampling for point cloud data.

    Args:
        dataset (io.Dataset): Point dataset.
        batch_size (int): batch size.
        shuffle (Optional[bool], optional): Whether to shuffle indices order before
            generating batch indices. Defaults to False.
        drop_last (Optional[bool], optional): Whether drop the last incomplete (less
            than 1 mini-batch) batch dataset. Defaults to False.
    """

    def __init__(
        self,
        dataset: io.Dataset,
        batch_size: int,
        shuffle: Optional[bool] = False,
        drop_last: Optional[bool] = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        super().__init__(dataset, None, shuffle, batch_size, drop_last)

    def __iter__(self):
        rest_points = misc.convert_to_array(
            self.dataset.input, tuple(self.dataset.input.keys())
        )
        rest_points_index = np.arange(len(rest_points))

        # shuffle manually
        if self.shuffle:
            rand_perm = np.random.permutation(len(rest_points))
            rest_points = rest_points[rand_perm]
            rest_points_index = rest_points_index[rand_perm]

        for batch_id in range(len(self)):
            batch_indices = []
            bary_center = np.mean(rest_points, axis=0)
            distance = np.full((len(rest_points),), np.nan)
            point = bary_center

            for point_i in range(min(self.batch_size, len(rest_points))):
                print(distance.shape, rest_points.shape)
                distance = np.minimum(
                    distance, np.sum((rest_points - point) ** 2, axis=1)
                )
                index = np.argmax(distance)

                original_index = rest_points_index[index]
                batch_indices.append(original_index)

                rest_points[index] = rest_points[-1]
                rest_points = rest_points[:-1]

                rest_points_index[index] = rest_points_index[-1]
                rest_points_index = rest_points_index[:-1]

                distance[index] = distance[-1]
                distance = distance[:-1]

            if len(batch_indices) == self.batch_size or not self.drop_last:
                yield batch_indices

            batch_indices = []


class DistributedFPSBatchSampler(io.DistributedBatchSampler):
    """Distributed batch sampler using Farthest Point Sampling for point cloud data.

    Args:
        dataset (io.Dataset): Point dataset.
        batch_size (int): batch size.
        shuffle (Optional[bool], optional): Whether to shuffle indices order before
            generating batch indices. Defaults to False.
        drop_last (Optional[bool], optional): Whether drop the last incomplete (less
            than 1 mini-batch) batch dataset. Defaults to False.
    """

    def __init__(
        self,
        dataset: io.Dataset,
        batch_size: int,
        shuffle: Optional[bool] = False,
        drop_last: Optional[bool] = False,
    ):
        super().__init__(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert (
            len(indices) == self.total_size
        ), f"len(indices){len(indices)} != self.total_size({self.total_size})"
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(
                self.local_rank * self.batch_size,
                len(indices) - last_batch_size,
                self.batch_size * self.nranks,
            ):
                subsampled_indices.extend(indices[i : i + self.batch_size])

            indices = indices[len(indices) - last_batch_size :]
            subsampled_indices.extend(
                indices[
                    self.local_rank
                    * last_local_batch_size : (self.local_rank + 1)
                    * last_local_batch_size
                ]
            )
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert (
            len(indices) == self.num_samples
        ), f"len(indices){len(indices)} != self.num_samples({self.num_samples})"

        local_rest_points = misc.convert_to_array(
            self.dataset.input, tuple(self.dataset.input.keys())
        )[indices]
        local_rest_points_index = np.arange(len(local_rest_points))

        for batch_id in range(len(self)):
            batch_indices = []
            bary_center = np.mean(local_rest_points, axis=0)
            distance = np.full((len(local_rest_points),), np.nan)
            point = bary_center

            for point_i in range(min(self.batch_size, len(local_rest_points))):
                print(distance.shape, local_rest_points.shape)
                distance = np.minimum(
                    distance, np.sum((local_rest_points - point) ** 2, axis=1)
                )
                index = np.argmax(distance)

                original_index = local_rest_points_index[index]
                batch_indices.append(original_index)

                local_rest_points[index] = local_rest_points[-1]
                local_rest_points = local_rest_points[:-1]

                local_rest_points_index[index] = local_rest_points_index[-1]
                local_rest_points_index = local_rest_points_index[:-1]

                distance[index] = distance[-1]
                distance = distance[:-1]

            if len(batch_indices) == self.batch_size or not self.drop_last:
                yield batch_indices

            batch_indices = []
