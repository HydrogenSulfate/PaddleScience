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

from ppsci.utils import misc


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
            self.dataset.input, ("x", "y")
        )  # [N, 2] 集合A的剩余点数
        # 记录原始集合A对应的下标（因为后面集合A是动态删点的，而返回的下标得是原始集合A的）
        rest_points_index = np.arange(len(rest_points))

        # shuffle manually
        if self.shuffle:
            rand_perm = np.random.permutation(len(rest_points))
            rest_points = rest_points[rand_perm]
            rest_points_index = rest_points_index[rand_perm]

        for batch_id in range(len(self)):
            # 每次返回batch个点，这batch个点是用最远点采样法得到的轮廓点
            batch_indices = []
            bary_center = np.mean(rest_points, axis=0)  # [3, ]得到集合A的重心点
            distance = np.full(
                (len(rest_points),), np.nan
            )  # 维护一个数组，表示集合A中的每个点到集合B的最短(平方)距离
            point = bary_center  # 虚拟初始点选为重心点，所以真正的第一个点是离重心最远的点

            for point_i in range(min(self.batch_size, len(rest_points))):
                print(distance.shape, rest_points.shape)
                distance = np.minimum(
                    distance, np.sum((rest_points - point) ** 2, axis=1)
                )  # 更新集合A到集合B的最短(平方)距离
                index = np.argmax(distance)  # 选出离集合B最远(平方距离最大)的那个点

                original_index = rest_points_index[index]
                batch_indices.append(original_index)

                # 使用比较高效的方式删除三个集合中index下标对应的数据
                rest_points[index] = rest_points[-1]
                rest_points = rest_points[:-1]

                rest_points_index[index] = rest_points_index[-1]
                rest_points_index = rest_points_index[:-1]

                distance[index] = distance[-1]
                distance = distance[:-1]

            if len(batch_indices) == self.batch_size or not self.drop_last:
                yield batch_indices

            batch_indices = []


# from ppsci.geometry import Rectangle
# from ppsci.constraint import InteriorConstraint
# from ppsci import loss
# geo = Rectangle([-5, -5], [5, 5])

# train_dataloader_cfg = {
#     "dataset": "NamedArrayDataset",
#     "iters_per_epoch": 5,
#     "sampler": {
#         "name": "BatchSampler",
#         "drop_last": True,
#         "shuffle": True,
#     },
#     "num_workers": 1,
#     "batch_size": 50
# }
# cst = InteriorConstraint(
#     {"z": lambda out: out["z"]},
#     {"z": 0},
#     geo,
#     train_dataloader_cfg,
#     loss.MSELoss(),
# )

# for input, label, weight in cst.data_loader:
#     print(input.keys())
