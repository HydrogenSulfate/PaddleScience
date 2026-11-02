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

import pytest
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer.lr import ExponentialDecay as PaddleExponentialDecay
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import PolynomialDecay

from ppsci.optimizer.lr_scheduler import Cosine
from ppsci.optimizer.lr_scheduler import ExponentialDecay
from ppsci.optimizer.lr_scheduler import Linear
from ppsci.optimizer.lr_scheduler import SchedulerList


def test_exponential_decay_by_iter():
    scheduler = ExponentialDecay(10, 10, 1e-3, gamma=0.95, decay_steps=3)()

    assert isinstance(scheduler, PaddleExponentialDecay)
    assert scheduler.by_epoch is False
    assert scheduler.get_lr() == pytest.approx(1e-3)

    scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-3 * 0.95 ** (1 / 3), rel=1e-6)

    scheduler.step()
    assert scheduler.get_lr() == pytest.approx(1e-3 * 0.95 ** (2 / 3), rel=1e-6)


def test_exponential_decay_by_epoch():
    scheduler = ExponentialDecay(
        10, 10, 1e-3, gamma=0.95, decay_steps=3, by_epoch=True
    )()

    assert scheduler.by_epoch is True
    assert scheduler.get_lr() == pytest.approx(1e-3)

    scheduler.step()
    # decay_steps is divided by iters_per_epoch in by_epoch mode.
    assert scheduler.get_lr() == pytest.approx(1e-3 * 0.95 ** (10 / 3), rel=1e-6)


def test_linear_decay():
    scheduler = Linear(10, 10, 1e-2)()

    assert isinstance(scheduler, PolynomialDecay)
    assert scheduler.by_epoch is False
    assert scheduler.get_lr() == pytest.approx(1e-2)

    scheduler.step()
    assert scheduler.get_lr() == pytest.approx(9.9e-3, rel=1e-6)

    scheduler.step()
    assert scheduler.get_lr() == pytest.approx(9.8e-3, rel=1e-6)


def test_cosine_decay():
    scheduler = Cosine(10, 10, 1e-3)()

    assert isinstance(scheduler, CosineAnnealingDecay)
    assert scheduler.get_lr() == pytest.approx(1e-3)

    lrs = []
    for _ in range(100):
        scheduler.step()
        lrs.append(scheduler.get_lr())

    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)


def test_cosine_decay_by_epoch():
    scheduler = Cosine(10, 10, 1e-3, by_epoch=True)()

    assert scheduler.by_epoch is True
    initial_lr = scheduler.get_lr()
    scheduler.step()
    assert scheduler.get_lr() < initial_lr


def test_linear_warmup_start():
    scheduler = Linear(10, 10, 1e-2, warmup_epoch=2, warmup_start_lr=1e-4)()

    assert isinstance(scheduler, LinearWarmup)
    assert scheduler.by_epoch is False
    assert scheduler.warmup_steps == 20
    assert scheduler.get_lr() == pytest.approx(1e-4)

    scheduler.step()
    expected = 1e-4 + (1e-2 - 1e-4) / 20
    assert scheduler.get_lr() == pytest.approx(expected, rel=1e-6)


def test_scheduler_list():
    linear = Linear(10, 5, 1e-2)()
    exponential = ExponentialDecay(10, 5, 1e-3, 0.9, 2)()
    scheduler_list = SchedulerList((linear, exponential))

    assert len(scheduler_list) == 2
    assert scheduler_list.by_epoch is False
    assert list(scheduler_list) == [linear, exponential]
    assert scheduler_list.get_lr() == pytest.approx(1e-2)

    scheduler_list.step()
    assert scheduler_list.get_lr() == pytest.approx(9.8e-3, rel=1e-6)


if __name__ == "__main__":
    pytest.main()
