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

import inspect
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

from hydra.core.config_store import ConfigStore
from hydra.core.object_type import ObjectType
from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import Literal

__all__ = ["SolverConfig", "apply_config"]


class EMAConfig(BaseModel):
    use_ema: bool = False
    decay: float = 0.9
    avg_freq: int = 1

    @field_validator("decay")
    def decay_check(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError(
                f"'ema.decay' should be in (0, 1) when is type of float, but got {v}"
            )
        return v

    @field_validator("avg_freq")
    def avg_freq_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'ema.avg_freq' should be a positive integer when is type of int, "
                f"but got {v}"
            )
        return v


class SWAConfig(BaseModel):
    use_swa: bool = False
    avg_freq: int = 1
    avg_range: Optional[Tuple[int, int]] = None

    @field_validator("avg_range")
    def avg_range_check(cls, v, info: ValidationInfo):
        if isinstance(v, tuple) and v[0] > v[1]:
            raise ValueError(f"'swa.avg_range' should be a valid range, but got {v}.")
        if isinstance(v, tuple) and v[0] < 0:
            raise ValueError(
                "The start epoch of 'swa.avg_range' should be a non-negative integer"
                f" , but got {v[0]}."
            )
        return v

    @field_validator("avg_freq")
    def avg_freq_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'swa.avg_freq' should be a positive integer when is type of int, "
                f"but got {v}"
            )
        return v


class TrainConfig(BaseModel):
    """
    Schema of training config for pydantic validation.
    """

    epochs: int = 1
    iters_per_epoch: int = 20
    update_freq: int = 1
    save_freq: int = 0
    eval_during_train: bool = False
    start_eval_epoch: int = 1
    eval_freq: int = 1
    checkpoint_path: Optional[str] = None
    pretrained_model_path: Optional[str] = None
    ema: Optional[EMAConfig] = None
    swa: Optional[SWAConfig] = None

    # Fine-grained validator(s) below
    @field_validator("epochs")
    def epochs_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'TRAIN.epochs' should be a positive integer when is type of int, "
                f"but got {v}"
            )
        return v

    @field_validator("iters_per_epoch")
    def iters_per_epoch_check(cls, v):
        if v <= 0 and v != -1:
            raise ValueError(
                f"'TRAIN.iters_per_epoch' received an invalid value({v}), "
                "but is expected one of: \n"
                "* A positive integer, to manually specify the number of iterations per epoch, "
                "which is commonly used in PINN training.\n"
                "* -1, to automatically set the number of iterations per epoch to "
                "the length of dataloader of given constraint, which is commonly "
                f"used in data-driven training.\n"
            )
        return v

    @field_validator("update_freq")
    def update_freq_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'TRAIN.update_freq' should be a positive integer when is type of int"
                f", but got {v}"
            )
        return v

    @field_validator("save_freq")
    def save_freq_check(cls, v):
        if v < 0:
            raise ValueError(
                "'TRAIN.save_freq' should be a non-negative integer when is type of int"
                f", but got {v}"
            )
        return v

    @field_validator("start_eval_epoch")
    def start_eval_epoch_check(cls, v, info: ValidationInfo):
        if info.data["eval_during_train"]:
            if v <= 0:
                raise ValueError(
                    f"'TRAIN.start_eval_epoch' should be a positive integer when "
                    f"'TRAIN.eval_during_train' is True, but got {v}"
                )
        return v

    @field_validator("eval_freq")
    def eval_freq_check(cls, v, info: ValidationInfo):
        if info.data["eval_during_train"]:
            if v <= 0:
                raise ValueError(
                    f"'TRAIN.eval_freq' should be a positive integer when "
                    f"'TRAIN.eval_during_train' is True, but got {v}"
                )
        return v

    @model_validator(mode="after")
    def ema_swa_checker(self):
        if (self.ema and self.swa) and (self.ema.use_ema and self.swa.use_swa):
            raise ValueError(
                "Cannot enable both EMA and SWA at the same time, "
                "please disable at least one of them."
            )
        return self

    @model_validator(mode="after")
    def swa_avg_range_checker(self):
        if self.swa and self.swa.use_swa and self.swa.avg_range[1] > self.epochs:
            raise ValueError(
                "The end epoch of 'swa.avg_range' should not be lager than "
                f"'epochs'({self.epochs}), but got {self.swa.avg_range[1]}."
            )
        return self


class EvalConfig(BaseModel):
    """
    Schema of evaluation config for pydantic validation.
    """

    pretrained_model_path: Optional[str] = None
    eval_with_no_grad: bool = False
    compute_metric_by_batch: bool = False
    batch_size: Optional[int] = 256

    @field_validator("batch_size")
    def batch_size_check(cls, v):
        if isinstance(v, int) and v <= 0:
            raise ValueError(
                f"'EVAL.batch_size' should be greater than 0 or None, but got {v}"
            )
        return v


class InferConfig(BaseModel):
    """
    Schema of inference config for pydantic validation.
    """

    pretrained_model_path: Optional[str] = None
    export_path: str = "./inference"
    pdmodel_path: Optional[str] = None
    pdiparams_path: Optional[str] = None
    onnx_path: Optional[str] = None
    device: Literal["cpu", "gpu", "npu", "xpu", "sdaa"] = "cpu"
    engine: Literal["native", "tensorrt", "onnx", "onednn"] = "native"
    precision: Literal["fp32", "fp16", "int8"] = "fp32"
    ir_optim: bool = True
    min_subgraph_size: int = 30
    gpu_mem: int = 2000
    gpu_id: int = 0
    max_batch_size: int = 1024
    num_cpu_threads: int = 10
    batch_size: Optional[int] = 256

    # Fine-grained validator(s) below
    @field_validator("engine")
    def engine_check(cls, v, info: ValidationInfo):
        if v == "mkldnn":
            raise ValueError(
                "The 'mkldnn' engine is deprecated. Please use 'onednn' instead."
            )
        if v == "tensorrt" and info.data["device"] != "gpu":
            raise ValueError(
                "'INFER.device' should be 'gpu' when 'INFER.engine' is 'tensorrt', "
                f"but got '{info.data['device']}'"
            )
        if v == "onednn" and info.data["device"] != "cpu":
            raise ValueError(
                "'INFER.device' should be 'cpu' when 'INFER.engine' is 'onednn', "
                f"but got '{info.data['device']}'"
            )

        return v

    @field_validator("min_subgraph_size")
    def min_subgraph_size_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'INFER.min_subgraph_size' should be greater than 0, " f"but got {v}"
            )
        return v

    @field_validator("gpu_mem")
    def gpu_mem_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'INFER.gpu_mem' should be greater than 0, " f"but got {v}"
            )
        return v

    @field_validator("gpu_id")
    def gpu_id_check(cls, v):
        if v < 0:
            raise ValueError(
                "'INFER.gpu_id' should be greater than or equal to 0, " f"but got {v}"
            )
        return v

    @field_validator("max_batch_size")
    def max_batch_size_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'INFER.max_batch_size' should be greater than 0, " f"but got {v}"
            )
        return v

    @field_validator("num_cpu_threads")
    def num_cpu_threads_check(cls, v):
        if v < 0:
            raise ValueError(
                "'INFER.num_cpu_threads' should be greater than or equal to 0, "
                f"but got {v}"
            )
        return v

    @field_validator("batch_size")
    def batch_size_check(cls, v):
        if isinstance(v, int) and v <= 0:
            raise ValueError(
                f"'INFER.batch_size' should be greater than 0 or None, but got {v}"
            )
        return v


class SolverConfig(BaseModel):
    """
    Schema of global config for pydantic validation.
    """

    # Global settings config
    mode: Union[Literal["train", "eval", "export", "infer"], str] = "train"
    output_dir: Optional[str] = None
    log_freq: int = 20
    seed: int = 42
    use_vdl: bool = False
    use_tbd: bool = False
    wandb_config: Mapping = {}
    use_wandb: bool = False
    device: Literal["cpu", "gpu", "xpu", "sdaa", None] = None
    use_amp: bool = False
    amp_level: Literal["O0", "O1", "O2", "OD"] = "O1"
    to_static: bool = False
    prim: bool = False
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    trace: bool = False

    # Training related config
    TRAIN: Optional[TrainConfig] = None

    # Evaluation related config
    EVAL: Optional[EvalConfig] = None

    # Inference related config
    INFER: Optional[InferConfig] = None

    # Fine-grained validator(s) below
    @field_validator("log_freq")
    def log_freq_check(cls, v):
        if v <= 0:
            raise ValueError(
                "'log_freq' should be a non-negative integer when is type of int"
                f", but got {v}"
            )
        return v

    @field_validator("seed")
    def seed_check(cls, v):
        if v < 0:
            raise ValueError(f"'seed' should be a non-negative integer, but got {v}")
        return v

    @field_validator("use_wandb")
    def use_wandb_check(cls, v, info: ValidationInfo):
        if v and not isinstance(info.data["wandb_config"], dict):
            raise ValueError(
                "'wandb_config' should be a dict when 'use_wandb' is True, "
                f"but got {info.data['wandb_config'].__class__.__name__}"
            )
        return v


_EXCLUDE_KEYS_DEFAULT = [
    "mode",
    "output_dir",
    "log_freq",
    "seed",
    "use_vdl",
    "use_tbd",
    "wandb_config",
    "use_wandb",
    "device",
    "use_amp",
    "amp_level",
    "to_static",
    "prim",
    "log_level",
    "trace",
    "TRAIN.save_freq",
    "TRAIN.eval_during_train",
    "TRAIN.start_eval_epoch",
    "TRAIN.eval_freq",
    "TRAIN.checkpoint_path",
    "TRAIN.pretrained_model_path",
    "EVAL.pretrained_model_path",
    "EVAL.eval_with_no_grad",
    "EVAL.compute_metric_by_batch",
    "EVAL.batch_size",
    "INFER.pretrained_model_path",
    "INFER.export_path",
    "INFER.pdmodel_path",
    "INFER.pdiparams_path",
    "INFER.onnx_path",
    "INFER.device",
    "INFER.engine",
    "INFER.precision",
    "INFER.ir_optim",
    "INFER.min_subgraph_size",
    "INFER.gpu_mem",
    "INFER.gpu_id",
    "INFER.max_batch_size",
    "INFER.num_cpu_threads",
    "INFER.batch_size",
]


def apply_config() -> None:
    """Register PaddleScience's Hydra defaults and OmegaConf resolvers explicitly.

    The package entry point, ``ppsci.__init__``, calls this function so existing
    examples can keep using ``import ppsci`` without registering anything
    themselves. Registration happens before Hydra composes a configuration;
    doing it in ``InitCallback`` or ``Solver`` would be too late.

    This module defines schemas without registering them at module scope. All
    PaddleScience config and resolver registrations are centralized here.

    This function is idempotent: existing config names and resolver names are
    preserved, including application-defined overrides. It can also restore
    registrations removed from ConfigStore or OmegaConf after an earlier call.

    Examples:
        >>> from ppsci.utils.config import apply_config
        >>> apply_config()
        >>> apply_config()  # Safe to call again.
    """
    cs = ConfigStore.instance()
    defaults = (
        (None, "ppsci_default", SolverConfig),
        ("TRAIN", "train_default", TrainConfig),
        ("TRAIN/ema", "ema_default", EMAConfig),
        ("TRAIN/swa", "swa_default", SWAConfig),
        ("EVAL", "eval_default", EvalConfig),
        ("INFER", "infer_default", InferConfig),
    )
    for group, name, schema in defaults:
        path = f"{group}/{name}.yaml" if group else f"{name}.yaml"
        if cs.get_type(path) == ObjectType.NOT_FOUND:
            cs.store(group=group, name=name, node=schema().model_dump())

    exclude_group = "hydra/job/config/override_dirname/exclude_keys"
    if (
        cs.get_type(f"{exclude_group}/exclude_keys_default.yaml")
        == ObjectType.NOT_FOUND
    ):
        cs.store(
            group=exclude_group,
            name="exclude_keys_default",
            node=_EXCLUDE_KEYS_DEFAULT,
        )

    # OmegaConf 2.4 promotes register_resolver to the modern API. On 2.2/2.3,
    # its legacy signature lacks "replace"; use register_new_resolver instead
    # so list arguments and nested interpolations keep their original types.
    if "replace" in inspect.signature(OmegaConf.register_resolver).parameters:
        register_resolver = OmegaConf.register_resolver
    else:
        register_resolver = OmegaConf.register_new_resolver

    if not OmegaConf.has_resolver("numpy"):
        import numpy as np

        register_resolver("numpy", lambda name: getattr(np, name))
    if not OmegaConf.has_resolver("sum"):
        register_resolver("sum", lambda values: sum(values))
