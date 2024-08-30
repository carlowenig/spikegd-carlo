import inspect
import shutil
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import EllipsisType
from typing import Any, Callable, Iterable, Literal, Self

import numpy as np
import yaml


class ConfigVariable(ABC):
    @abstractmethod
    def get_values(self, config: dict, var_name: str) -> list[Any]: ...

    # def get_values_from_config(self, config: dict, var_name: str) -> list[Any]:
    #     try:
    #         args = filter_dict(
    #             config,
    #             self.dependencies,
    #             require_all=True,
    #         )
    #     except ValueError as e:
    #         if self.on_missing_dependency == "raise":
    #             raise ValueError(
    #                 f"Missing dependencies for variable '{var_name}'"
    #             ) from e
    #         else:
    #             return []

    #     try:
    #         return self.get_values(**args)
    #     except Exception as e:
    #         raise ValueError(
    #             f"Error while computing values of variable '{var_name}'"
    #         ) from e


class FixedValuesConfigVariable(ConfigVariable):
    def __init__(self, values: list[Any]):
        self.values = values

    def get_values(self, config: dict, var_name: str) -> list[Any]:
        return self.values


class ComputedConfigVariable(ConfigVariable):
    def __init__(
        self,
        compute: Callable[..., list],
        on_missing_dependency: Literal["raise", "ignore"] = "raise",
        dependencies: Iterable[str] | None = None,
    ):
        self.compute = compute
        self.dependencies = list(
            dependencies
            if dependencies is not None
            else inspect.signature(compute).parameters.keys()
        )
        self.on_missing_dependency = on_missing_dependency

    def get_values(self, config: dict, var_name: str) -> list[Any]:
        try:
            args = filter_dict(
                config,
                self.dependencies,
                require_all=True,
            )
        except ValueError as e:
            if self.on_missing_dependency == "raise":
                raise ValueError(
                    f"Missing dependencies for variable '{var_name}'"
                ) from e
            else:
                return []

        try:
            return self.compute(**args)
        except Exception as e:
            raise ValueError(
                f"Error while computing values of variable '{var_name}' with args {args}"
            ) from e


def vary(*values):
    values = list(values)
    return FixedValuesConfigVariable(values)


def computed(
    compute: Callable[..., Any],
    on_missing_dependency: Literal["raise", "ignore"] = "raise",
):
    return ComputedConfigVariable(
        lambda **kwargs: [compute(**kwargs)],
        on_missing_dependency,
        dependencies=inspect.signature(compute).parameters.keys(),
    )


def computed_vary(
    compute: Callable[..., list],
    on_missing_dependency: Literal["raise", "ignore"] = "raise",
):
    return ComputedConfigVariable(compute, on_missing_dependency)


def filter_dict[K, V](
    d: dict[K, V],
    predicate_or_keys: Callable[[K, V], bool] | Iterable[K],
    *,
    require_all=False,
) -> dict[K, V]:
    if isinstance(predicate_or_keys, Iterable):
        keys = predicate_or_keys
        result = {k: v for k, v in d.items() if k in keys}

        if require_all:
            missing_keys = set(keys) - set(result.keys())
            if missing_keys:
                raise ValueError(
                    f"Missing keys: {", ".join(map(str, missing_keys))}\n"
                    f"Available keys: {", ".join(map(str, d.keys()))}"
                )

        return result
    else:
        predicate = predicate_or_keys
        return {k: v for k, v in d.items() if predicate(k, v)}


def expand_config(
    config: dict | type | list[dict | type],
) -> list[dict[str, Any]]:
    configs = []

    if isinstance(config, list):
        # Argument is a list -> expand each element
        for c in config:
            configs.extend(expand_config(c))
    elif isinstance(config, type):
        return expand_config(dict(config.__dict__))  # type: ignore
    elif ... in config:
        # Argument contains ellipsis -> expand each variation
        variations = config.pop(...)
        for v in variations:
            configs.extend(expand_config(config | v))
    else:
        # Argument is a single config -> expand ConfigVariable values
        constants: dict[str, Any] = {}
        variables: dict[str, ConfigVariable] = {}

        for key, value in config.items():
            assert not isinstance(key, EllipsisType)

            if isinstance(value, ConfigVariable):
                variables[key] = value
            else:
                constants[key] = value

        configs.append(constants)

        for key, var in variables.items():
            available_configs = configs.copy()

            # print(f"Computing {key} with {len(available_configs)} configs")

            for c in available_configs:
                values = var.get_values(c, key)
                if len(values) == 0:
                    continue  # No values possible -> skip

                # Add first value to current config
                c[key] = values[0]
                # Create new configs for all other values
                for value in values[1:]:
                    configs.append(c | {key: value})

    return configs


def get_constants(configs: list[dict]) -> tuple[dict[str, Any], list[str]]:
    assert isinstance(configs, list), f"Invalid config list: {configs}"

    if len(configs) == 0:
        return {}, []

    assert isinstance(configs[0], dict), f"Invalid config: {configs[0]}"
    constants = configs[0].copy()
    varying_keys = []

    for config in configs[1:]:
        assert isinstance(config, dict), f"Invalid config: {config}"
        for key, value in list(constants.items()):
            if config.get(key) != value:
                constants.pop(key)
                varying_keys.append(key)

    return constants, varying_keys


def format_timestamp(t: float):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S_%f")


def parse_timestamp(t: str):
    return datetime.strptime(t, "%Y-%m-%d_%H-%M-%S_%f").timestamp()


def get_grid_data_path(version: int, root: str | Path):
    return Path(root) / f"grid_data_{version:02d}"


_DETECTABLE_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}


def detect_constant(
    x: float, min_power=-1, max_power=1, max_int=100
) -> tuple[str | None, int, Fraction]:
    from fractions import Fraction

    if x == 0:
        return None, 0, Fraction(0)

    for name, value in _DETECTABLE_CONSTANTS.items():
        for power in range(min_power, max_power + 1):
            if power == 0:
                continue

            frac = Fraction(x / value**power)
            if frac.numerator < max_int and frac.denominator < max_int:
                return name, power, frac

    return None, 0, Fraction(0)


def format_number(x: float, f_str: str = ".3g"):
    constant, power, coeff = detect_constant(x)

    if constant is not None:
        return f"{coeff ** power} {constant}^{power}"

    return f"{x:{f_str}}"


@dataclass
class GridData:
    version: int
    info: dict
    root: str | Path = "results"
    trials: dict[int, "GridTrial"] = field(default_factory=dict)

    def get_path(self):
        return get_grid_data_path(self.version, self.root)

    def save(self):
        path = self.get_path()
        path.mkdir(parents=True, exist_ok=True)
        info_path = path / "info.yaml"
        trials_path = path / "trials"

        with open(info_path, "w") as f:
            yaml.dump(self.info, f, sort_keys=False)

        for trial in self.trials.values():
            trial_path = trials_path / f"trial_{trial.index:03d}.yaml"
            trial.save(trial_path, if_exists="override")

    @classmethod
    def load_or_create(
        cls, version: int, root: str | Path = "results", backup=True
    ) -> Self:
        path = get_grid_data_path(version, root)

        if not path.exists():
            info = {
                "created_at": format_timestamp(time.time()),
            }
            data = cls(version, info, root)
            data.save()
            return data

        if backup:
            shutil.copytree(
                path,
                path.with_name(
                    f"{path.stem}_backup_{time.strftime("%Y-%m-%d_%H-%M-%S")}"
                ),
            )

        info_path = path / "info.yaml"
        trials_path = path / "trials"

        if not info_path.exists():
            raise FileNotFoundError(f"GridData info file {info_path} not found.")

        with info_path.open("r") as f:
            info = yaml.safe_load(f)

        if trials_path.exists():
            trials = {
                int(p.stem.split("_")[1]): GridTrial.load(p)
                for p in trials_path.glob("trial_*.yaml")
            }
        else:
            trials = {}

        return cls(version, info, root, trials)

    def get_next_trial_index(self):
        return max((trial.index for trial in self.trials.values()), default=0) + 1

    def start_trial(self, config: dict[str, Any]) -> "GridTrial":
        trial = GridTrial(
            index=self.get_next_trial_index(),
            config=config,
            started_at=format_timestamp(time.time()),
        )
        self.trials[trial.index] = trial
        return trial

    def find_trial_by_config(self, config: dict[str, Any]):
        for trial in self.trials.values():
            if trial.config == config:
                return trial

    def get_mean_duration(self):
        return np.mean(
            [
                trial.duration
                for trial in self.trials.values()
                if trial.duration is not None
            ]
        ).item()


@dataclass
class GridTrial:
    index: int
    config: dict[str, Any]
    started_at: str
    finished_at: str | None = None
    duration: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self):
        if self.index > 999:
            raise ValueError("Trial indices above 999 are not supported.")

    def to_dict(self):
        return asdict(self)

    def save(
        self,
        path: Path,
        if_exists: Literal["compare", "raise", "skip", "override"] = "override",
    ):
        if path.exists():
            if if_exists == "raise":
                raise FileExistsError(f"File {path} already exists.")
            elif if_exists == "skip":
                return
            elif if_exists == "override":
                pass
            elif if_exists == "compare":
                trial = GridTrial.load(path)
                if trial != self:
                    raise ValueError(
                        f"Existing trial data in {path} does not match new trial data.\n"
                        f"Existing: {trial}\n"
                        f"New: {self}"
                    )
            else:
                raise ValueError(f"Invalid value for if_exists: {if_exists}")

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"GridTrial file {path} not found.")

        with path.open("r") as f:
            dict_ = yaml.safe_load(f)

        try:
            return cls(**dict_)
        except TypeError as e:
            raise ValueError(
                f"Error while loading trial from {path}. Consider starting a new grid data version."
            ) from e

    @classmethod
    def from_item(cls, item: tuple[int, dict[str, Any]]):
        index, dict_ = item
        assert isinstance(index, int), f"Invalid trial index: {index}"
        assert isinstance(dict_, dict), f"Invalid trial data: {dict_}"

        trial = cls(**dict_)
        assert trial.index == index, f"Trial index mismatch: {trial.index} != {index}"
        return trial

    def finish(self, metrics: dict[str, Any], error: str | None = None):
        self.metrics = metrics
        self.error = error
        end_time = time.time()
        self.finished_at = format_timestamp(end_time)
        self.duration = end_time - parse_timestamp(self.started_at)

    def restart(self):
        self.started_at = format_timestamp(time.time())
        self.finished_at = None
        self.duration = None
        self.metrics = {}
        self.error = None


def print_dict(d: dict, value_format="", indent=26):
    for k, v in d.items():
        if isinstance(v, float):
            v_str = format_number(v, value_format)
        else:
            v_str = f"{v:{value_format}}"

        # indent
        v_str = v_str.replace("\n", "\n" + " " * indent)

        print(f"{k:<{indent - 1}} {v_str}")


def scan_grid(
    func: Callable[[dict[str, Any]], dict[str, Any]],
    config_grid: dict,
    version: int,
    show_metrics: Iterable[str] = (),
    if_trial_exists: Literal[
        "skip", "recompute", "recompute_if_error", "raise", "warn"
    ] = "recompute_if_error",
):
    configs = expand_config(config_grid)

    constants, varying_keys = get_constants(configs)

    data = GridData.load_or_create(version)
    print_dict(
        {
            "varying keys": ", ".join(varying_keys),
            "configs": len(configs),
        }
    )
    print("========== CONSTANTS ==========")
    print_dict(constants)
    print()

    print()

    for config_index, config in enumerate(configs):
        print(f"========== CONFIG {config_index + 1:03d} ==========")
        print_dict(filter_dict(config, varying_keys))

        # Check if this config has already been run
        trial = data.find_trial_by_config(config)

        if trial is not None:
            if if_trial_exists == "raise":
                raise ValueError(
                    f"This config has already been used in trial {trial.index}."
                )
            elif if_trial_exists == "warn":
                warnings.warn(
                    f"This config has already been used in trial {trial.index}."
                )
                continue
            elif if_trial_exists == "ignore":
                print(
                    f"This config has already been used in trial {trial.index}. Skipping."
                )
                continue
            elif if_trial_exists == "recompute":
                print(
                    f"This config has already been used in trial {trial.index}. Recomputing."
                )
                trial.restart()
            elif if_trial_exists == "recompute_if_error":
                if trial.error is None:
                    print(
                        f"This config has already been used in trial {trial.index} and had no error. Skipping."
                    )
                    continue
                else:
                    print(
                        f"This config has already been used in trial {trial.index} but had an error. Recomputing."
                    )
                    trial.restart()
        else:
            trial = data.start_trial(config)
            print(f"Starting trial {trial.index}.")

        try:
            metrics = func(config)
        except Exception:
            e_str = traceback.format_exc()
            trial.finish({}, e_str)
            print_dict({"error": e_str})
        else:
            trial.finish(metrics)
            print_dict(filter_dict(metrics, show_metrics, require_all=True))

        data.save()

        print()
        remaining_configs = len(configs) - config_index - 1
        estimated_remaining_time = remaining_configs * data.get_mean_duration()
        eta = datetime.fromtimestamp(time.time() + estimated_remaining_time)
        print(
            f"Remaining: {remaining_configs} configs "
            f"({estimated_remaining_time:.1f}s, ETA: {eta.strftime('%H:%M:%S')})"
        )
        print()
