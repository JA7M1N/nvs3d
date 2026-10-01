"""Configuration dataclasses, YAML serialization, and CLI override utilities."""

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar
import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base configuration shared by all experiments."""

    exp_name: str = "default"
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "runs"
    iters: int = 100
    log_every: int = 10
    save_every: int = 1000
    kind: str = "base"

    # Generic dictionary for extra parameters
    extra: dict[str, Any] = field(default_factory=dict)


def save_config_yaml(config: Any, path: Path | str) -> None:
    """Serialize configuration dataclass or dictionary to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(config, "__dataclass_fields__"):
        data = asdict(config)
    elif isinstance(config, dict):
        data = config
    else:
        raise TypeError(f"Expected dataclass or dict, got {type(config)}")

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_config_yaml(path: Path | str, config_cls: type[T] = BaseConfig) -> T:
    """Load configuration YAML file into a typed dataclass instance."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    # Extract fields recognized by config_cls
    cls_field_names = {f.name for f in fields(config_cls)}
    init_kwargs: dict[str, Any] = {}
    extra_kwargs: dict[str, Any] = {}

    for k, v in raw_dict.items():
        if k in cls_field_names:
            init_kwargs[k] = v
        else:
            extra_kwargs[k] = v

    if "extra" in cls_field_names and extra_kwargs:
        init_kwargs["extra"] = extra_kwargs

    return config_cls(**init_kwargs)


def _cast_value(val_str: str) -> Any:
    """Cast a string value to bool, int, float, or keep as string."""
    lower = val_str.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def parse_cli_overrides(config: T, cli_args: list[str]) -> T:
    """Parse list of CLI arguments (e.g. ['--exp_name', 'foo', '--iters=200']) and override config fields."""
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]
        if arg.startswith("--"):
            key_val = arg[2:]
            if "=" in key_val:
                key, val_str = key_val.split("=", 1)
                overrides[key] = _cast_value(val_str)
                i += 1
            else:
                key = key_val
                if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                    overrides[key] = _cast_value(cli_args[i + 1])
                    i += 2
                else:
                    overrides[key] = True
                    i += 1
        else:
            i += 1

    valid_field_names = {f.name for f in fields(config)}
    updated_kwargs: dict[str, Any] = {}
    extra = dict(getattr(config, "extra", {}))

    for k, v in overrides.items():
        if k in valid_field_names:
            updated_kwargs[k] = v
        else:
            extra[k] = v

    if "extra" in valid_field_names:
        updated_kwargs["extra"] = extra

    return replace(config, **updated_kwargs)
