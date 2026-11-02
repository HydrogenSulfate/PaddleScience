# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import pytest

import ppsci
from ppsci import arch
from ppsci.loss import mtl

__all__ = []


@pytest.fixture
def model():
    return arch.MLP(("x", "y"), ("u", "v"), 3, 16)


def _check_agg_state_dict(agg, model):
    model_state = model.state_dict()
    agg_state = agg.state_dict()
    for key in agg_state:
        assert key not in model_state


def test_AGDA(model):
    aggregator = mtl.AGDA(model)
    assert aggregator.should_persist is False


def test_GradNorm(model):
    aggregator = mtl.GradNorm(model)
    assert aggregator.should_persist is True
    _check_agg_state_dict(aggregator, model)


def test_PCGrad(model):
    aggregator = mtl.PCGrad(model)
    assert aggregator.should_persist is False


def test_Relobralo():
    aggregator = mtl.Relobralo(num_losses=2)
    assert aggregator.should_persist is True
    assert dict(aggregator.state_dict()) != {}


def test_Sum():
    aggregator = mtl.Sum()
    assert aggregator.should_persist is False
    assert dict(aggregator.state_dict()) == {}


def test_NTK(model):
    aggregator = mtl.NTK(model)
    assert aggregator.should_persist is True
    _check_agg_state_dict(aggregator, model)


def test_restore_aggregator(tmp_path):
    model = arch.MLP(["x", "y"], ["u"], 2, 16)
    optimizer = ppsci.optimizer.Adam(1e-3)(model)
    equation = ppsci.equation.Laplace(2)
    geom = ppsci.geometry.Rectangle([0, 0], [1, 1])
    bc = ppsci.constraint.BoundaryConstraint(
        equation.equations,
        {"laplace": 0.0},
        geom,
        {
            "dataset": "IterableNamedArrayDataset",
            "iters_per_epoch": 1,
            "batch_size": 16,
        },
        loss=ppsci.loss.MSELoss(),
        name="bound",
    )
    solver = ppsci.solver.Solver(
        model,
        {"bound": bc},
        output_dir=str(tmp_path),
        optimizer=optimizer,
        iters_per_epoch=1,
        epochs=1,
        log_freq=1,
        device="cpu",
    )
    solver.train()

    # Resume from the checkpoint written above; this only needs to load, not train.
    ppsci.solver.Solver(
        model,
        {"bound": bc},
        output_dir=str(tmp_path),
        optimizer=optimizer,
        iters_per_epoch=1,
        epochs=1,
        device="cpu",
        checkpoint_path=str(tmp_path / "checkpoints" / "latest"),
    )


if __name__ == "__main__":
    pytest.main()
