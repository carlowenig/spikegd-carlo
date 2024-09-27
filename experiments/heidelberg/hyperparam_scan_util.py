import hashlib
import inspect
import json
import os
import shutil
import time
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import repeat
from pathlib import Path
from types import EllipsisType
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Literal,
    Mapping,
    Self,
    Sequence,
    Set,
    Unpack,
    final,
)

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from spikegd.utils.formatting import (
    fmt_dict_multiline,
    fmt_duration,
    fmt_timestamp,
    parse_timestamp,
)


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


PathLike = str | Path | None


def as_path(path: PathLike) -> Path:
    if isinstance(path, Path):
        return path
    elif path is None:
        return Path(".")
    else:
        return Path(path)


@dataclass
class Resource[*Args](ABC):
    root: PathLike = field(kw_only=True)

    @classmethod
    @abstractmethod
    def get_rel_path_of(cls, *args: Unpack[Args]) -> Path: ...

    @classmethod
    @abstractmethod
    def create(cls, *args: Unpack[Args], root: PathLike) -> Self: ...

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

        obj._snapshot = obj.copy()

        return obj

    @classmethod
    def load_or_create(
        cls,
        *args: Unpack[Args],
        root: PathLike,
        **create_kwargs,
    ) -> Self:
        try:
            return cls.load(*args, root=root)
        except FileNotFoundError:
            return cls.create(*args, root=root, **create_kwargs)

    def get_rel_path(self) -> Path:
        args = self._get_args()
        return type(self).get_rel_path_of(*args)

    def get_abs_path(self) -> Path:
        return as_path(self.root) / self.get_rel_path()

    def exists(self):
        return self.get_abs_path().exists()

    def to_dict(self) -> dict[str, Any]:
        dict_ = asdict(self)
        del dict_["root"]
        return dict_

    def copy(self, *, remove_snapshot=True):
        copy = deepcopy(self)
        if remove_snapshot:
            copy._snapshot = None
        return copy

    @classmethod
    def from_dict(cls, dict_: dict[str, Any], root: PathLike) -> Self:
        return cls(root=root, **dict_)

    @final
    def save(
        self,
        if_exists: Literal["compare", "raise", "skip", "override"] = "override",
    ):
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
        self._snapshot = self.copy()

    @classmethod
    def delete_by_args(cls, *args: Unpack[Args], root: PathLike):
        path = cls.get_abs_path_of(*args, root=root)

        if not path.exists():
            return
        elif path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

    def delete(self):
        type(self).delete_by_args(*self._get_args(), root=self.root)
        self._snapshot = None

    def __str__(self):
        return f"{type(self).__name__}({", ".join(str(a) for a in self._get_args())})"


class FolderWithInfoYamlResource[*Args](Resource[*Args]):
    _info_file_path: ClassVar[str] = "info.yaml"

    @classmethod
    def _load_from(cls, abs_path: Path, root: PathLike) -> Self:
        if not abs_path.exists():
            raise FileNotFoundError(f"{cls.__name__} at {abs_path} not found.")

        info_path = abs_path / cls._info_file_path

        with info_path.open("r") as f:
            dict_ = yaml.load(f, yaml.Loader)

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
            path.with_name(f"{path.name}_backup_{time.strftime("%Y%m%d_%H%M%S")}"),
        )


class YamlResource[*Args](Resource[*Args]):
    @classmethod
    def _load_from(cls, abs_path: Path, root: PathLike) -> Self:
        if not abs_path.exists():
            raise FileNotFoundError(f"{cls.__name__} at {abs_path} not found.")

        with abs_path.open("r") as f:
            dict_ = yaml.load(f, yaml.Loader)

        return cls.from_dict(dict_, root=root)

    def _save_to_unchecked(self, abs_path: Path):
        with abs_path.open("w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)


def _load_trial_dict(args: tuple[str, PathLike, str]) -> dict:
    scan_id, scan_root, config_hash = args

    with GridTrial.get_abs_path_of(scan_id, config_hash, root=scan_root).open("r") as f:
        trial_dict = yaml.load(f, yaml.Loader)

    assert isinstance(trial_dict, dict)

    config = trial_dict.pop("config")
    assert isinstance(config, dict)

    for key, value in config.items():
        if key == "_index":
            continue
        trial_dict[f"config.{key}"] = value

    metrics = trial_dict.pop("metrics")
    assert isinstance(metrics, dict)

    for key, value in metrics.items():
        if key == "epochs":
            continue
        trial_dict[f"metrics.{key}"] = value

    return trial_dict


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
    ):
        scan = cls(
            id=id,
            created_at=fmt_timestamp(time.time()),
            metadata=metadata or {},
            root=root,
        )
        scan.save()
        return scan

    def _get_args(self):
        return (self.id,)

    def load_run_ids(self) -> list[str]:
        runs_path = self.get_abs_path() / "runs"

        if not runs_path.exists():
            return []

        return [
            p.name
            for p in runs_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]

    def load_run(self, id: str):
        return GridRun.load(self.id, id, root=self.root)

    def load_runs(self):
        return {run_id: self.load_run(run_id) for run_id in self.load_run_ids()}

    def load_trial_config_hashes(self) -> list[str]:
        trials_path = self.get_abs_path() / "trials"

        if not trials_path.exists():
            return []

        return [p.stem for p in trials_path.iterdir() if p.is_file()]

    def load_trial(self, config_hash: str):
        return GridTrial.load(self.id, config_hash, root=self.root)

    def load_trials(self, *, progress=False):
        if progress:
            print(f"Loading trials for scan {self.id}...")
        return {
            config_hash: self.load_trial(config_hash)
            for config_hash in tqdm(
                self.load_trial_config_hashes(),
                disable=not progress,
            )
        }

    def load_trials_df(
        self, *, progress=False, config_hashes: list[str] | None = None, parallel=True
    ):
        if config_hashes is None:
            config_hashes = self.load_trial_config_hashes()

        if parallel:
            import multiprocessing

            args_iter = zip(repeat(self.id), repeat(self.root), config_hashes)

            with multiprocessing.Pool() as pool:
                if progress:
                    trial_dicts = list(
                        tqdm(
                            pool.imap_unordered(_load_trial_dict, args_iter),
                            total=len(config_hashes),
                        )
                    )
                else:
                    trial_dicts = pool.map(_load_trial_dict, args_iter)
        else:
            trial_dicts = [
                _load_trial_dict((self.id, self.root, config_hash))
                for config_hash in tqdm(config_hashes, disable=not progress)
            ]

        return pd.DataFrame(trial_dicts).sort_values("started_at")

    def sync_trials_df(
        self, path: PathLike, *, verbose=False, parallel=True
    ) -> pd.DataFrame:
        path = self.get_abs_path() / as_path(path)

        config_hashes = self.load_trial_config_hashes()

        df = pd.read_parquet(path) if path.exists() else None

        if df is not None:
            if verbose:
                print(f"Found DataFrame with {len(df)} trials.")

            for existing_hash in list(df["config_hash"]):
                if existing_hash not in config_hashes:
                    # Remove trials that are not in the scan anymore
                    df = df[df["config_hash"] != existing_hash]
                else:
                    # Skip trials that are already in the dataframe
                    config_hashes.remove(existing_hash)

        if len(config_hashes) == 0:
            if verbose:
                print("Trials DataFrame is up-to-date.")

            if df is None:
                df = pd.DataFrame()
        else:
            if verbose:
                print(f"Loading {len(config_hashes)} new trials...")

            new_df = self.load_trials_df(
                progress=verbose, config_hashes=config_hashes, parallel=parallel
            )

            if df is None:
                df = new_df
            else:
                df = pd.concat([df, new_df], ignore_index=True)

        df.sort_values("started_at", inplace=True, ignore_index=True)

        df.to_parquet(path)

        if verbose:
            print(f"Saved DataFrame with {len(df)} trials to {path}.")

        return df

    def load_exported_trials_df(self, path: PathLike = None):
        if path is None:
            paths = self.get_abs_path().glob("trials_export_*.parquet")
            path = max(paths, key=os.path.getctime)
        else:
            path = self.get_abs_path() / path

        return pd.read_parquet(path)

    def get_mean_trial_duration(self):
        return np.mean(
            [
                trial.duration
                for trial in self.load_trials().values()
                if trial.duration is not None
            ]
        ).item()

    def clean(self, *, delete_unsuccessful_trials=True, delete_empty_runs=True):
        trials = self.load_trials()

        if delete_unsuccessful_trials:
            for trial_id, trial in trials.copy().items():
                if trial.error is not None:
                    print(f"Deleting unsuccessful trial {trial_id}")
                    trial.delete()
                    del trials[trial_id]

        if delete_empty_runs:
            non_empty_run_ids = {trial.run_id for trial in trials.values()}
            empty_run_ids = set(self.load_run_ids()) - non_empty_run_ids

            for run_id in empty_run_ids:
                print(f"Deleting empty run {run_id}")
                self.load_run(run_id).delete()

    def update_trials(
        self,
        func: Callable[["GridTrial"], "GridTrial | None | Literal[False]"]
        | None = None,
        verbose=True,
    ):
        old_config_hashes = self.load_trial_config_hashes()
        updated_count = 0
        failed_count = 0
        unchanged_count = 0
        skipped_count = 0

        pbar = tqdm(old_config_hashes, disable=not verbose)

        for old_config_hash in pbar:
            old_trial = self.load_trial(old_config_hash)
            new_trial = old_trial.copy()
            skip = False

            if func is not None:
                result = func(new_trial)
                if result is None:
                    pass
                elif result is False:
                    # Skip this trial
                    skip = True
                elif isinstance(result, GridTrial):
                    new_trial = result
                else:
                    raise ValueError(
                        f"Invalid return value from trial update function: {result}"
                    )

            new_trial.update_config_hash()

            if skip:
                skipped_count += 1
            elif new_trial == old_trial:
                unchanged_count += 1
            else:
                try:
                    old_trial.delete()
                except Exception as e:
                    raise ValueError(
                        f"Could not delete old trial {old_trial.config_hash}"
                    ) from e

                try:
                    new_trial.save(if_exists="raise")
                except Exception as e:
                    failed_count += 1
                    if verbose:
                        print(
                            f"Error while saving updated trial {old_config_hash} as "
                            f"new trial {new_trial.config_hash}:\n"
                            f"  {str(e).replace("\n", "\n  ")}\n"
                            "Reverting to old trial..."
                        )
                    # Revert to old trial
                    old_trial.save(if_exists="raise")
                else:
                    updated_count += 1

            pbar.set_postfix(
                updated=updated_count,
                failed=failed_count,
                unchanged=unchanged_count,
                skipped=skipped_count,
            )

        if verbose:
            print(
                f"Updated: {updated_count}\n"
                f"Failed: {failed_count}\n"
                f"Unchanged: {unchanged_count}\n"
                f"Skipped: {skipped_count}"
            )

    def run(
        self,
        func: Callable[[dict[str, Any]], dict[str, Any]],
        config_grid: dict,
        *,
        show_metrics: Iterable[str] = (),
        if_trial_exists: Literal[
            "skip", "recompute", "recompute_if_error", "raise"
        ] = "recompute_if_error",
        base_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        n_jobs: int | None = None,
        job_index: int = 1,  # 1-based index
        preview=False,
    ):
        if base_id is None:
            base_id = GridRun.create_timestamp_id(name)

        configs = expand_config(config_grid)
        n_configs_total = len(configs)

        if n_jobs is not None:
            if n_jobs > n_configs_total:
                print(
                    f"WARNING: Using more jobs {n_jobs} than there are configs to "
                    f"process ({n_configs_total})!"
                )

            n_configs_per_job = int(np.ceil(n_configs_total / n_jobs))
            config_start = (job_index - 1) * n_configs_per_job
            config_end = min(len(configs), job_index * n_configs_per_job)
            configs = configs[config_start:config_end]

            n_digit = len(str(n_jobs))
            full_id = f"{base_id}_{job_index:0{n_digit}d}"
        else:
            config_start = 0
            config_end = len(configs)
            full_id = base_id

        if len(configs) == 0:
            print("No configs to process. Skipping run.")
            return

        constants, variables = get_constants(configs)

        if preview:
            if n_jobs is not None:
                print(f"JOB {job_index} / {n_jobs}")

            print(
                f"PROCESSING {len(configs)} of {n_configs_total} CONFIGS "
                f"(index {config_start} - {config_end - 1})"
            )
            print(
                f"CONSTANTS:\n  {fmt_dict_multiline(constants).replace("\n", "\n  ")}"
            )
            print(f"VARIABLES: {", ".join(variables)}")
            return

        run = GridRun.create(
            self.id,
            full_id,
            root=self.root,
            name=name,
            description=description,
            constants=constants,
            variables=variables,
            metadata={
                "base_id": base_id,
                "split_size": n_jobs,
                "split_index": job_index,
            },
        )
        print(
            f"Started run {run.id}. Log can be found at grid_scans/runs/{run.id}/main.log"
        )

        if n_jobs is not None:
            run.log(f"JOB {job_index} / {n_jobs}")

        run.log(
            f"PROCESSING {len(configs)} of {n_configs_total} CONFIGS "
            f"(index {config_start} - {config_end - 1})"
        )
        run.log(f"CONSTANTS:\n  {fmt_dict_multiline(constants).replace("\n", "\n  ")}")
        run.log(f"VARIABLES: {", ".join(variables)}")

        # run.log("Loading trials for duration calculations...")
        # cached_trials = self.load_trials()
        # run.log(f"Loaded {len(cached_trials)} trials.")
        cached_trials: dict[str, GridTrial] = {}

        def process_config(config_index: int, config: dict):
            config_hash = hash_config(config)

            remaining_configs = len(configs) - config_index
            mean_duration = np.mean(
                [trial.duration for trial in cached_trials.values()]
            ).item()

            if not np.isnan(mean_duration):
                remaining_time = remaining_configs * mean_duration
                remaining_time_str = fmt_duration(remaining_time)
                eta = datetime.fromtimestamp(time.time() + remaining_time).strftime(
                    "%H:%M:%S"
                )
                run.log(
                    f"{remaining_configs} configs remaining "
                    f"(~{remaining_time_str}, ETA: {eta}, "
                    f"based on a mean trial duration of {mean_duration:.1f}s)"
                )
            else:
                run.log(
                    f"{remaining_configs} configs remaining "
                    f"(could not estimate remaining time)"
                )
            print()

            run.log(
                f"========== CONFIG {config_index + config_start:03d} ==========\n"
                + fmt_dict_multiline(filter_dict(config, variables))
            )

            # Check if this config has already been run in this scan
            run.log("Checking for existing trial...")
            try:
                existing_trial = self.load_trial(config_hash)
            except FileNotFoundError:
                # config hash does not exist yet -> continue
                pass
            else:
                # config hash already exists
                if existing_trial.config != config:
                    raise ValueError(
                        f"Hash collision for config hash {config_hash}:\n"
                        f"  Config 1: {config}\n"
                        f"  Config 2: {existing_trial.config}"
                    )

                if if_trial_exists == "raise":
                    raise ValueError(
                        f"This config has already been used in trial {config_hash}."
                    )
                elif if_trial_exists == "skip":
                    run.log(
                        f"This config has already been used in trial {config_hash}. "
                        "Skipping."
                    )
                    cached_trials[config_hash] = existing_trial
                    return
                elif if_trial_exists == "recompute":
                    run.log(
                        f"This config has already been used in trial {config_hash}. "
                        "Recomputing."
                    )
                    existing_trial.delete()
                elif if_trial_exists == "recompute_if_error":
                    if existing_trial.error is None:
                        run.log(
                            f"This config has already been used in trial {config_hash} "
                            "and finished successfully. Skipping."
                        )
                        cached_trials[config_hash] = existing_trial
                        return
                    else:
                        run.log(
                            f"This config has already been used in trial {config_hash} "
                            "but did not finish successfully. Recomputing."
                        )
                        existing_trial.delete()
                else:
                    raise ValueError(
                        f"Invalid value for if_trial_exists: {if_trial_exists}"
                    )

            run.log(f"Starting trial {config_hash}.")

            started_time = time.time()

            try:
                metrics = func(config)
            except Exception:
                error = traceback.format_exc()
                metrics = {}
                run.log(fmt_dict_multiline({"error": error}))
            else:
                error = None
                run.log(
                    fmt_dict_multiline(
                        filter_dict(metrics, show_metrics, require_all=True)
                    )
                )

            finished_time = time.time()

            try:
                trial = GridTrial.create(
                    self.id,
                    config_hash,
                    run_id=run.id,
                    config=config,
                    root=self.root,
                    started_time=started_time,
                    finished_time=finished_time,
                    metrics=metrics,
                    error=error,
                )
            except Exception as e:
                run.log(f"Error while saving trial {config_hash}: {e}")
            else:
                cached_trials[config_hash] = trial
                run.log(f"Finished trial {config_hash} after {trial.duration:.1f}s")

        try:
            # TODO: Is it possible to parallelize this?
            # import jax.numpy as jnp

            # arr_configs = {
            #     "_index": jnp.arange(len(configs)),
            # } | {k: jnp.array([c[k] for c in configs]) for k in configs[0].keys()}

            # jax.pmap(process_config)(arr_configs)

            # with ThreadPool(n_processes) as pool:
            #     pool.map(process_config, enumerate(configs))

            for config_index, config in enumerate(configs):
                process_config(config_index, config)

        finally:
            run.finished_at = fmt_timestamp(time.time())
            run.duration = time.time() - parse_timestamp(run.started_at)
            run.save()
            print(f"Finished run {self.id}")

        return run


@dataclass
class GridRun(FolderWithInfoYamlResource[str, str]):
    scan_id: str
    id: str
    name: str | None
    description: str | None
    metadata: dict
    constants: dict[str, Any]
    variables: list[str]
    started_at: str
    finished_at: str | None = None
    duration: float | None = None

    @classmethod
    def get_rel_path_of(cls, scan_id: str, run_id: str):
        return GridScan.get_rel_path_of(scan_id) / "runs" / run_id

    @classmethod
    def create_timestamp_id(cls, name: str | None = None):
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
        constants: dict[str, Any] | None = None,
        variables: list[str] | None = None,
    ):
        if id is None:
            id = cls.create_timestamp_id(name)

        assert constants is not None
        assert variables is not None

        run = cls(
            root=root,
            scan_id=scan_id,
            id=id,
            name=name,
            description=description,
            metadata=metadata or {},
            constants=constants,
            variables=variables,
            started_at=fmt_timestamp(time.time()),
        )
        run.save()
        return run

    def _get_args(self):
        return (
            self.scan_id,
            self.id,
        )

    def log(self, message: str):
        # thread_id = threading.get_ident()
        log_path = self.get_abs_path() / "main.log"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with log_path.open("a") as file:
            file.write(f"[{timestamp}]\n{message}\n")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    elif isinstance(value, (bool, int, float, np.number)):
        return float(value)
    elif isinstance(value, Mapping):
        return {
            str(_normalize_value(k)): _normalize_value(value[k]) for k in sorted(value)
        }
    elif isinstance(value, Set):
        return {_normalize_value(v) for v in value}
    elif isinstance(value, Sequence):
        return [_normalize_value(v) for v in value]
    else:
        return value


def hash_config(config: dict[str, Any]):
    config_str = json.dumps(_normalize_value(config), sort_keys=True)
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()


@dataclass
class GridTrial(YamlResource[str, str]):
    root: PathLike
    scan_id: str
    run_id: str | None
    config: dict[str, Any]
    started_at: str
    finished_at: str
    duration: float
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    config_hash: str = ""

    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = hash_config(self.config)

    def update_config_hash(self):
        self.config_hash = hash_config(self.config)

    @classmethod
    def get_rel_path_of(cls, scan_id: str, config_hash: str):
        return GridScan.get_rel_path_of(scan_id) / "trials" / f"{config_hash}.yaml"

    @classmethod
    def create(
        cls,
        scan_id: str,
        config_hash: str = "",
        *,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        started_time: float | None = None,
        finished_time: float | None = None,
        root: PathLike,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        if_exists: Literal["compare", "raise", "skip", "override"] = "raise",
    ):
        assert started_time is not None
        assert finished_time is not None

        trial = cls(
            root=root,
            scan_id=scan_id,
            run_id=run_id,
            config=config.copy() if config is not None else {},
            config_hash=config_hash,
            started_at=fmt_timestamp(started_time),
            finished_at=fmt_timestamp(finished_time),
            duration=finished_time - started_time,
            metrics=metrics if metrics is not None else {},
            error=error,
        )
        trial.save(if_exists=if_exists)
        return trial

    def _get_args(self):
        return (self.scan_id, self.config_hash)
