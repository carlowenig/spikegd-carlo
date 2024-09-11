import inspect
import shutil
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from fractions import Fraction
from math import isnan
from pathlib import Path
from types import EllipsisType
from typing import Any, Callable, ClassVar, Iterable, Literal, Self, Unpack, final

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


PathLike = str | Path | None


def as_path(path: PathLike) -> Path:
    if isinstance(path, Path):
        return path
    elif path is None:
        return Path(".")
    else:
        return Path(path)


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
class Resource[*Args](ABC):
    root: PathLike = field(kw_only=True)
    owner: str = field(kw_only=True)

    @classmethod
    @abstractmethod
    def get_rel_path_of(cls, *args: Unpack[Args]) -> Path: ...

    @classmethod
    @abstractmethod
    def create(cls, *args: Unpack[Args], root: PathLike, author: str) -> Self: ...

    @classmethod
    @abstractmethod
    def _load_from(cls, abs_path: Path, root: PathLike) -> Self: ...

    @abstractmethod
    def _save_to_unchecked(self, abs_path: Path) -> None: ...

    @abstractmethod
    def _get_args(self) -> tuple[*Args]: ...

    @classmethod
    @final
    def get_abs_path_of(cls, *args: Unpack[Args], root: PathLike) -> Path:
        return as_path(root) / cls.get_rel_path_of(*args)

    @classmethod
    @final
    def load(cls, *args: Unpack[Args], root: PathLike) -> Self:
        abs_path = cls.get_abs_path_of(*args, root=root)
        obj = cls._load_from(abs_path, root=root)

        if obj._get_args() != args:
            raise ValueError(
                f"Loaded {cls.__name__} has different arguments: {obj._get_args()} != {args}"
            )

        return obj

    @classmethod
    def load_or_create(
        cls,
        *args: Unpack[Args],
        root: PathLike,
        author: str,
        **create_kwargs,
    ) -> Self:
        try:
            return cls.load(*args, root=root)
        except FileNotFoundError:
            return cls.create(*args, root=root, author=author, **create_kwargs)

    def get_rel_path(self) -> Path:
        args = self._get_args()
        return type(self).get_rel_path_of(*args)

    def get_abs_path(self) -> Path:
        return as_path(self.root) / self.get_rel_path()

    def exists(self):
        return self.get_abs_path().exists()

    def require_modification(self, author: str):
        if self.owner != author:
            raise PermissionError(
                f"Resource at {self.get_abs_path()} is owned by {self.owner!r} and cannot be modified by {author!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        dict_ = asdict(self)
        del dict_["root"]
        return dict_

    @classmethod
    def from_dict(cls, dict_: dict[str, Any], root: PathLike) -> Self:
        return cls(root=root, **dict_)

    @final
    def save(
        self,
        author: str,
        if_exists: Literal["compare", "raise", "skip", "override"] = "override",
    ):
        self.require_modification(author)

        path = self.get_abs_path()

        if path.exists():
            if if_exists == "raise":
                raise FileExistsError(
                    f"{type(self).__name__} at {path} already exists."
                )
            elif if_exists == "skip":
                return
            elif if_exists == "override":
                pass
            elif if_exists == "compare":
                existing = type(self)._load_from(path, root=self.root)
                if existing != self:
                    raise ValueError(
                        f"Existing {type(self).__name__} at {path} does not match new one.\n"
                        f"Existing: {existing}\n"
                        f"New: {self}"
                    )
            else:
                raise ValueError(f"Invalid value for if_exists: {if_exists}")

        path.parent.mkdir(parents=True, exist_ok=True)

        self._save_to_unchecked(path)

    def delete(self, author: str):
        self.require_modification(author)

        path = self.get_abs_path()

        if not path.exists():
            return
        elif path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)


class FolderWithInfoYamlResource[*Args](Resource[*Args]):
    _info_file_path: ClassVar[str] = "info.yaml"

    @classmethod
    def _load_from(cls, abs_path: Path, root: PathLike) -> Self:
        if not abs_path.exists():
            raise FileNotFoundError(f"{cls.__name__} at {abs_path} not found.")

        info_path = abs_path / cls._info_file_path

        with info_path.open("r") as f:
            dict_ = yaml.safe_load(f)

        return cls.from_dict(dict_, root=root)

    def _save_to_unchecked(self, abs_path: Path):
        abs_path.mkdir(parents=True, exist_ok=True)
        info_path = abs_path / self._info_file_path

        with open(info_path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    def backup(self):
        path = self.get_abs_path()
        shutil.copytree(
            path,
            path.with_name(f"{path.stem}_backup_{time.strftime("%Y%m%d_%H%M%S")}"),
        )


class YamlResource[*Args](Resource[*Args]):
    @classmethod
    def _load_from(cls, abs_path: Path, root: PathLike) -> Self:
        if not abs_path.exists():
            raise FileNotFoundError(f"{cls.__name__} at {abs_path} not found.")

        with abs_path.open("r") as f:
            dict_ = yaml.safe_load(f)

        return cls.from_dict(dict_, root=root)

    def _save_to_unchecked(self, abs_path: Path):
        with abs_path.open("w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)


@dataclass
class GridScan(FolderWithInfoYamlResource[str]):
    id: str
    created_at: str
    metadata: dict = field(default_factory=dict)
    # runs: dict[str, "GridRun"] = field(default_factory=dict)

    @classmethod
    def get_rel_path_of(cls, id: str):
        return Path("grid_scans") / id

    @classmethod
    def create(
        cls,
        id: str,
        *,
        metadata: dict | None = None,
        root: PathLike,
        author: str,
    ):
        scan = cls(
            id=id,
            created_at=format_timestamp(time.time()),
            metadata=metadata or {},
            root=root,
            owner=author,
        )
        scan.save(author)
        return scan

    def _get_args(self):
        return (self.id,)

    def create_run(
        self, *, name: str | None = None, description: str | None = None, author: str
    ):
        return GridRun.create(
            self.id,
            root=self.root,
            author=author,
            name=name,
            description=description,
        )

    def load_run_ids(self) -> list[str]:
        runs_path = self.get_abs_path() / "runs"

        if not runs_path.exists():
            return []

        return [
            p.name
            for p in runs_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]

    def load_runs(self):
        return {
            run_id: GridRun.load(self.id, run_id, root=self.root)
            for run_id in self.load_run_ids()
        }

    def load_trials(self):
        return {
            (run.id, trial.index): trial
            for run in self.load_runs().values()
            for trial in run.load_trials().values()
        }

    def find_trial_by_config(self, config: dict[str, Any]):
        for trial in self.load_trials().values():
            if trial.config == config:
                return trial

    def get_mean_trial_duration(self):
        return np.mean(
            [
                trial.duration
                for trial in self.load_trials().values()
                if trial.duration is not None
            ]
        ).item()

    def clean(
        self, author: str, *, delete_unsuccessful_trials=True, delete_empty_runs=True
    ):
        for run in self.load_runs().values():
            run.clean(
                author,
                delete_unsuccessful_trials=delete_unsuccessful_trials,
                delete_if_empty=delete_empty_runs,
            )


@dataclass
class GridRun(FolderWithInfoYamlResource[str, str]):
    scan_id: str
    id: str
    name: str | None
    description: str | None
    metadata: dict
    started_at: str | None = None
    finished_at: str | None = None
    duration: float | None = None

    @classmethod
    def get_rel_path_of(cls, scan_id: str, run_id: str):
        return GridScan.get_rel_path_of(scan_id) / "runs" / run_id

    @classmethod
    def create_id(cls, name: str | None = None):
        id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name is not None:
            id += f"_{name}"
        return id

    @classmethod
    def create(
        cls,
        scan_id: str,
        id: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
        root: PathLike,
        author: str,
    ):
        if id is None:
            id = cls.create_id(name)

        run = cls(
            root=root,
            owner=author,
            scan_id=scan_id,
            id=id,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        run.save(author)
        return run

    def _get_args(self):
        return (
            self.scan_id,
            self.id,
        )

    def load_trial_indices(self) -> list[int]:
        trials_path = self.get_abs_path() / "trials"

        if not trials_path.exists():
            return []

        return [
            int(p.stem)
            for p in trials_path.iterdir()
            if p.is_file() and p.stem.isdigit()
        ]

    def load_next_trial_index(self):
        return max(self.load_trial_indices(), default=-1) + 1

    def load_trials(self):
        return {
            index: GridTrial.load(self.scan_id, self.id, index, root=self.root)
            for index in self.load_trial_indices()
        }

    def create_trial(self, config: dict[str, Any], *, author: str):
        index = self.load_next_trial_index()
        return GridTrial.create(
            self.scan_id,
            self.id,
            index,
            config=config,
            author=author,
            root=self.root,
        )

    def clean(
        self, author: str, *, delete_unsuccessful_trials=True, delete_if_empty=True
    ):
        trials = self.load_trials()

        if delete_unsuccessful_trials:
            for trial_id, trial in trials.copy().items():
                if not trial.has_finished_successfully:
                    trial.delete(author)
                    del trials[trial_id]

        if delete_if_empty and len(trials) == 0:
            self.delete(author)

    def run(
        self,
        func: Callable[[dict[str, Any]], dict[str, Any]],
        config_grid: dict,
        *,
        show_metrics: Iterable[str] = (),
        if_trial_exists: Literal[
            "skip", "recompute", "recompute_if_unsuccessful", "raise", "warn"
        ] = "recompute_if_unsuccessful",
        max_configs: int | None = None,
        author: str,
    ):
        if self.started_at is not None:
            raise ValueError(f"Run {self.id} has already been started.")

        self.require_modification(author)
        self.started_at = format_timestamp(time.time())
        self.save(author)

        try:
            configs = expand_config(config_grid)

            constants, varying_keys = get_constants(configs)

            scan = GridScan.load_or_create(self.scan_id, root=self.root, author=author)

            print_dict(
                {
                    "varying keys": ", ".join(varying_keys),
                }
            )
            print("========== CONSTANTS ==========")
            print_dict(constants)

            print()

            for config_index, config in enumerate(configs):
                if max_configs is not None and config_index == max_configs:
                    print("Reached maximum number of configs.")
                    break

                remaining_configs = len(configs) - config_index
                mean_duration = scan.get_mean_trial_duration()

                if not isnan(mean_duration):
                    remaining_time = remaining_configs * mean_duration
                    remaining_time_str = (
                        f"{int(remaining_time) // 60}m" f"{int(remaining_time) % 60}s"
                    )
                    eta = datetime.fromtimestamp(time.time() + remaining_time).strftime(
                        "%H:%M:%S"
                    )
                    print(
                        f"{remaining_configs} configs remaining "
                        f"(~{remaining_time_str}, ETA: {eta}, "
                        f"based on a mean trial duration of {mean_duration:.1f}s)"
                    )
                else:
                    print(
                        f"{remaining_configs} configs remaining "
                        f"(could not estimate remaining time)"
                    )
                print()

                print(f"========== CONFIG {config_index + 1:03d} ==========")
                print_dict(filter_dict(config, varying_keys))

                # Check if this config has already been run in this scan
                existing_trial = scan.find_trial_by_config(config)

                if existing_trial is not None:
                    if if_trial_exists == "raise":
                        raise ValueError(
                            f"This config has already been used in run {existing_trial.run_id}, trial {existing_trial.index}."
                        )
                    elif if_trial_exists == "warn":
                        warnings.warn(
                            f"This config has already been used in run {existing_trial.run_id}, trial {existing_trial.index}."
                        )
                        continue
                    elif if_trial_exists == "skip":
                        print(
                            f"This config has already been used in run {existing_trial.run_id}, trial {existing_trial.index}. "
                            "Skipping."
                        )
                        print()
                        continue
                    elif if_trial_exists == "recompute":
                        print(
                            f"This config has already been used in run {existing_trial.run_id}, trial {existing_trial.index}. "
                            "Recomputing."
                        )
                        # trial.restart(author=author)
                    elif if_trial_exists == "recompute_if_unsuccessful":
                        if existing_trial.has_finished_successfully:
                            print(
                                f"This config has already been used in run {existing_trial.run_id}, trial {existing_trial.index} "
                                "and finished successfully. Skipping."
                            )
                            print()
                            continue
                        else:
                            print(
                                f"This config has already been used in run {existing_trial.run_id}, trial {existing_trial.index} "
                                "but did not finish successfully. Recomputing."
                            )
                            # trial.restart(author=author)
                    else:
                        raise ValueError(
                            f"Invalid value for if_trial_exists: {if_trial_exists}"
                        )

                trial = self.create_trial(config, author=author)
                print(f"Starting trial {trial.index}.")

                try:
                    metrics = func(config)
                except Exception:
                    e_str = traceback.format_exc()
                    trial.finish({}, e_str, author=author)
                    print_dict({"error": e_str})
                else:
                    trial.finish(metrics, author=author)
                    print_dict(filter_dict(metrics, show_metrics, require_all=True))

                print()

        finally:
            self.finished_at = format_timestamp(time.time())
            self.duration = time.time() - parse_timestamp(self.started_at)
            self.save(author)
            print(f"Finished run {self.id}")


@dataclass
class GridTrial(YamlResource[str, str, int]):
    root: PathLike
    scan_id: str
    run_id: str
    index: int
    config: dict[str, Any]
    started_at: str
    finished_at: str | None = None
    duration: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self):
        if self.index > 9999:
            raise ValueError("Trial indices above 9999 are not supported.")

    @classmethod
    def get_rel_path_of(cls, scan_id: str, run_id: str, index: int):
        return GridRun.get_rel_path_of(scan_id, run_id) / "trials" / f"{index:04d}.yaml"

    @classmethod
    def create(
        cls,
        scan_id: str,
        run_id: str,
        index: int,
        *,
        config: dict[str, Any] | None = None,
        root: PathLike,
        author: str,
    ):
        trial = cls(
            root=root,
            owner=author,
            scan_id=scan_id,
            run_id=run_id,
            index=index,
            config=config.copy() if config is not None else {},
            started_at=format_timestamp(time.time()),
        )
        trial.save(author)
        return trial

    def _get_args(self):
        return (self.scan_id, self.run_id, self.index)

    def finish(self, metrics: dict[str, Any], error: str | None = None, *, author: str):
        self.require_modification(author)
        end_time = time.time()
        self.metrics = metrics
        self.error = error
        self.finished_at = format_timestamp(end_time)
        self.duration = end_time - parse_timestamp(self.started_at)
        self.save(author)

    def restart(self, author: str):
        self.require_modification(author)
        self.started_at = format_timestamp(time.time())
        self.finished_at = None
        self.duration = None
        self.metrics = {}
        self.error = None
        self.save(author)

    @property
    def has_finished(self):
        return self.finished_at is not None

    @property
    def has_finished_successfully(self):
        return self.has_finished and self.error is None


def print_dict(d: dict, value_format="", indent=26):
    for k, v in d.items():
        if isinstance(v, float):
            v_str = format_number(v, value_format)
        else:
            v_str = f"{v:{value_format}}"

        # indent
        v_str = v_str.replace("\n", "\n" + " " * indent)

        print(f"{k:<{indent - 1}} {v_str}")
