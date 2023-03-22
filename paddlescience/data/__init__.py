# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import types

import paddle
import paddle.distributed as dist

from paddlescience.data import trphysx_dataset

from .data_process import load_data
from .data_process import save_data


def build_dataloader(
    dataset_name, batch_size, shuffle, drop_last, num_workers=8, dataset_args=dict()
):
    """
    Build the dataloader according to arguments
    dataset_name - name of the dataset
    batch_size - batch size,
    shuffle - is shuffle data
    drop_last - whether drop the last incomplete batch dataset size
            is not divisible by the batch size.
    """
    dataset_args = types.MappingProxyType(dataset_args)

    if dataset_name == "LorenzDataset":
        dataset = trphysx_dataset.LorenzDataset(**dataset_args)
    elif dataset_name == "CylinderDataset":
        dataset = trphysx_dataset.CylinderDataset(**dataset_args)
    elif dataset_name == "RosslerDataset":
        dataset = trphysx_dataset.RosslerDataset(**dataset_args)
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented.")

    if dist.get_world_size() > 1:
        # Distribute data to multiple cards
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )
    else:
        # Distribute data to single card
        batch_sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
        )

    data_loader = paddle.io.DataLoader(
        dataset=dataset, batch_sampler=batch_sampler, num_workers=num_workers
    )

    return data_loader
