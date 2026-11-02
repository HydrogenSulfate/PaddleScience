import copy
import inspect
import subprocess
import sys
from pathlib import Path

import hydra
import pytest
import yaml
from hydra.core.config_store import ConfigStore
from hydra.core.object_type import ObjectType
from omegaconf import OmegaConf

from ppsci.utils import config as config_module
from ppsci.utils.callbacks import InitCallback


@pytest.fixture
def isolated_config_registries():
    """Restore process-global Hydra and OmegaConf state after each config test."""
    cs = ConfigStore.instance()
    repo = copy.deepcopy(cs.repo)
    resolvers = {name: OmegaConf._get_resolver(name) for name in ("numpy", "sum")}
    yield cs
    cs.repo = repo
    for name, resolver in resolvers.items():
        OmegaConf.clear_resolver(name)
        if resolver is not None:
            # OmegaConf exposes no public API for restoring a wrapped resolver.
            from omegaconf.basecontainer import BaseContainer

            BaseContainer._resolvers[name] = resolver


def test_package_import_applies_config():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from hydra.core.config_store import ConfigStore; "
            "from omegaconf import OmegaConf; "
            "assert not OmegaConf.has_resolver('sum'); "
            "assert not OmegaConf.has_resolver('numpy'); "
            "import ppsci; "
            "assert ConfigStore.instance().load('ppsci_default.yaml').node.seed == 42; "
            "assert OmegaConf.create({'x': '${sum:[1, 2, 3]}'}).x == 6; "
            "assert OmegaConf.has_resolver('numpy')",
        ],
        check=True,
    )


def test_reload_config_does_not_register_defaults_or_resolvers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import copy, importlib, sys; "
            "from ppsci.utils import config; "
            "from hydra.core.config_store import ConfigStore; "
            "from omegaconf import OmegaConf; "
            "cs = ConfigStore.instance(); "
            "del cs.repo['ppsci_default.yaml']; "
            "OmegaConf.clear_resolver('sum'); "
            "OmegaConf.clear_resolver('numpy'); "
            "before = copy.deepcopy(cs.repo); "
            "importlib.reload(config); "
            "assert cs.repo == before; "
            "assert not OmegaConf.has_resolver('numpy'); "
            "assert not OmegaConf.has_resolver('sum'); "
            "assert 'ppsci.utils.logger' not in sys.modules; "
            "assert 'paddle' not in sys.modules; "
            "config.SolverConfig(); "
            "assert cs.repo == before",
        ],
        check=True,
    )


def test_apply_config_registers_defaults_idempotently(
    isolated_config_registries, monkeypatch
):
    cs = isolated_config_registries
    config_module.apply_config()
    defaults = {
        "ppsci_default.yaml": config_module.SolverConfig,
        "TRAIN/train_default.yaml": config_module.TrainConfig,
        "TRAIN/ema/ema_default.yaml": config_module.EMAConfig,
        "TRAIN/swa/swa_default.yaml": config_module.SWAConfig,
        "EVAL/eval_default.yaml": config_module.EvalConfig,
        "INFER/infer_default.yaml": config_module.InferConfig,
    }
    for path, schema in defaults.items():
        node = cs.load(path).node
        assert OmegaConf.to_container(node) == schema().model_dump()

    excluded = cs.load(
        "hydra/job/config/override_dirname/exclude_keys/exclude_keys_default.yaml"
    ).node
    assert "TRAIN.save_freq" in excluded
    assert "INFER.batch_size" in excluded

    def fail_on_store(*args, **kwargs):
        pytest.fail("Repeated apply_config() must not rewrite registered configs")

    monkeypatch.setattr(cs, "store", fail_on_store)
    config_module.apply_config()


def test_apply_config_resolvers_support_lists_and_nested_interpolations(
    isolated_config_registries,
):
    config_module.apply_config()
    cfg = OmegaConf.create(
        {
            "values": [10, 20, 30],
            "total": "${sum:${values}}",
            "literal_total": "${sum:[1, 2, 3]}",
            "pi": "${numpy:pi}",
        }
    )
    assert cfg.total == 60
    assert cfg.literal_total == 6
    assert cfg.pi == pytest.approx(3.141592653589793)


def test_apply_config_preserves_application_registrations(isolated_config_registries):
    cs = isolated_config_registries
    cs.store(name="ppsci_default", node={"seed": 123})
    OmegaConf.clear_resolver("sum")
    if "replace" in inspect.signature(OmegaConf.register_resolver).parameters:
        register_resolver = OmegaConf.register_resolver
    else:
        register_resolver = OmegaConf.register_new_resolver
    register_resolver("sum", lambda values: 99)

    config_module.apply_config()

    assert cs.load("ppsci_default.yaml").node.seed == 123
    assert OmegaConf.create({"total": "${sum:[1, 2]}"}).total == 99


def test_apply_config_restores_removed_registrations(isolated_config_registries):
    cs = isolated_config_registries
    config_module.apply_config()
    del cs.repo["ppsci_default.yaml"]
    OmegaConf.clear_resolver("sum")
    OmegaConf.clear_resolver("numpy")

    config_module.apply_config()

    assert cs.get_type("ppsci_default.yaml") == ObjectType.CONFIG
    assert OmegaConf.has_resolver("sum")
    assert OmegaConf.has_resolver("numpy")


def test_package_import_supports_hydra_compose():
    config_dir = Path(__file__).resolve().parents[2] / "examples/allen_cahn/conf"
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="allen_cahn_piratenet.yaml")
    assert cfg.seed == 42
    assert cfg.TRAIN.update_freq == 1
    assert cfg.TRAIN.ema.use_ema is False
    assert cfg.TRAIN.swa.use_swa is False
    assert cfg.INFER.gpu_id == 0


@pytest.mark.parametrize(
    "epochs,mode,seed",
    [
        (-1, "train", 1024),
        (10, "eval", -1),
    ],
)
def test_invalid_epochs(tmp_path, epochs, mode, seed):
    cfg_dict = {
        "hydra": {
            "callbacks": {
                "init_callback": {"_target_": "ppsci.utils.callbacks.InitCallback"}
            }
        },
        "mode": mode,
        "seed": seed,
        "TRAIN": {
            "epochs": epochs,
        },
    }

    (tmp_path / "test_config.yaml").write_text(yaml.dump(cfg_dict))

    with hydra.initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        cfg = hydra.compose(config_name="test_config.yaml")

        with pytest.raises(SystemExit) as exec_info:
            InitCallback().on_job_start(config=cfg)
        assert exec_info.value.code == 2


# 这部分通常不需要，除非你想直接从脚本运行测试
if __name__ == "__main__":
    pytest.main()
