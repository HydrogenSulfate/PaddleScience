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

from __future__ import annotations

from typing import List
from typing import Literal

import numpy as np
import paddle
from einops import rearrange


def shuffle_along_axis(a, axis):
    """Shuffle numpy array along the specified axis."""
    a = np.swapaxes(a, axis, 0)  # 把要 shuffle 的 axis 移到前面
    np.random.shuffle(a)  # 沿第0维打乱
    a = np.swapaxes(a, 0, axis)  # 再换回来
    return a


class TMTDataset(paddle.io.Dataset):
    # Whether support batch indexing for speeding up fetching process.
    batch_index: bool = True

    def __init__(
        self,
        input_keys,
        label_keys,
        data_path: str,
        num_train: int,
        mode: Literal["train", "test"] = "train",
    ):
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.data_path = data_path
        self.num_train = num_train
        self.mode = mode
        train_outputs, test_outputs, mean, std = self._get_dataset(data_path)
        train_outputs = rearrange(train_outputs, "b h w c -> (b c) h w")  # [b4, h, w]
        test_outputs = rearrange(test_outputs, "b h w c -> (b c) h w")  # [b4, h, w]

        self.train_outputs = train_outputs[..., None]  # [b4, h, w, 1]
        self.test_outputs = test_outputs[..., None]  # [b4, h, w, 1]

    def __getitem__(self, idx: int | List[int]) -> np.ndarray:
        if self.mode == "train":
            sample_or_batch = self.train_outputs[idx]
        else:
            sample_or_batch = self.test_outputs[idx]
        return sample_or_batch

    def _get_dataset(self, data_path):
        data = np.load(data_path, allow_pickle=True).item()

        u = data["x_velocity"]
        v = data["y_velocity"]
        p = data["pressure"]
        # udm = data["udm"]
        sdf = data["sdf"]

        outputs = np.stack([u, v, p, sdf], axis=-1)  # (b, h, w, c)

        # Shuffle dataset
        outputs = shuffle_along_axis(outputs, axis=0)
        outputs = np.array(outputs, dtype=np.float32)

        train_outputs = outputs[: self.num_train]  # [n_train, h, w, 4]
        test_outputs = outputs[self.num_train :]  # [n_test, h, w, 4]

        # Normalize the data 对数据进行z-score标准化
        mean = train_outputs.mean(axis=(0, 1, 2))  # [4]
        std = train_outputs.std(axis=(0, 1, 2))  # [4]

        train_outputs = (train_outputs - mean) / std  # [n_train, h, w, 4]
        test_outputs = (test_outputs - mean) / std  # [n_test, h, w, 4]

        return train_outputs, test_outputs, mean, std

    def __len__(self):
        if self.mode == "train":
            return len(self.train_outputs)
        else:
            return len(self.test_outputs)


class BatchParser:
    def __init__(self, num_queries, h, w, solution):
        self.num_query_points = num_queries
        self.solution = solution

        x_star = np.linspace(0, 1, h)
        y_star = np.linspace(0, 1, w)
        x_star, y_star = np.meshgrid(x_star, y_star, indexing="ij")

        self.coords = np.hstack(
            [x_star.flatten()[:, None], y_star.flatten()[:, None]]
        ).astype(
            paddle.get_default_dtype()
        )  # (h * w, 2)

    def random_query(self, batch: paddle.Tensor, downsample: int = 1):
        batch_inputs = batch  # [b, h, w, 1]
        b, h, w, c = batch.shape
        # batch_outputs = rearrange(batch, "b h w c -> b (h w) c") # [b, hw, 1]
        batch_outputs = paddle.reshape(batch, [b, -1, c])  # [b, hw, 1]

        query_index = np.random.choice(
            batch_outputs.shape[1], size=(self.num_query_points,), replace=False
        )  # [num_query_points]

        batch_coords = self.coords[query_index][None, ...]  # [num_query_points, 2]
        batch_outputs = batch_outputs[:, query_index]  # [b, num_query_points, 1]

        # Downsample the inputs
        if len(self.solution) == 1:
            batch_inputs = batch_inputs[:, ::downsample, ::downsample]  # [b, h', w', 1]
        else:
            sol = np.random.choice(self.solution)
            batch_inputs = batch_inputs[:, ::sol, ::sol]  # [b, h', w', 1]

        # batch_coords: [1, num_query_points, 2]
        # batch_inputs: [b, h', w', 1]
        # batch_outputs: [b, num_query_points, 1]
        return (
            {
                "coords": paddle.to_tensor(batch_coords),
                "x": batch_inputs,
                "u": batch_outputs,
            },
            {
                "u": batch_outputs,
            },
            None,
        )

    def query_all(self, batch):
        batch_inputs = batch

        batch_outputs = rearrange(batch, "b h w c -> b (h w) c")
        batch_coords = self.coords

        return batch_coords, batch_inputs, batch_outputs
