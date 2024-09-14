import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Self


@dataclass
class Dependency:
    no_default: ClassVar = object()

    name: str
    default: Any = no_default

    @classmethod
    def from_param(cls, param: inspect.Parameter):
        if param.default is inspect.Parameter.empty:
            default = Dependency.no_default
        else:
            default = param.default

        return cls(param.name, default)


def infer_dependencies(func):
    return [
        Dependency.from_param(p) for p in inspect.signature(func).parameters.values()
    ]


class ConfigVariable[T](ABC):
    @abstractmethod
    def get_values(self, config: dict, var_name: str) -> list[T]: ...


class FixedConfigVariable[T](ConfigVariable[T]):
    def __init__(self, value: T):
        self.value = value

    def get_values(self, config: dict, var_name: str) -> list[T]:
        return [self.value]


def fixed[T](value: T) -> FixedConfigVariable[T]:
    return FixedConfigVariable(value)


class ChoiceConfigVariable[T](ConfigVariable[T]):
    def __init__(self, values: Iterable[T]):
        self.values = list(values)

    def get_values(self, config: dict, var_name: str) -> list[T]:
        return self.values


def choice[T](*values: T) -> ChoiceConfigVariable[T]:
    return ChoiceConfigVariable(values)


class ComputedConfigVariable[T](ConfigVariable[T]):
    def __init__(
        self,
        compute: Callable[..., T | ConfigVariable[T]],
        dependencies: Iterable[Dependency] | None = None,
    ):
        self.compute = compute
        self.dependencies = (
            infer_dependencies(compute) if dependencies is None else list(dependencies)
        )

    def get_values(self, config: dict, var_name: str) -> list[T]:
        args = {}

        for dep in self.dependencies:
            if dep.name in config:
                value = config[dep.name]
            elif dep.default is not Dependency.no_default:
                value = dep.default
            else:
                raise ValueError(
                    f"Missing dependency '{dep.name}' for variable '{var_name}'"
                )

            args[dep.name] = value

        try:
            result = self.compute(**args)
        except Exception as e:
            raise ValueError(
                f"Error while computing {var_name}"
                f"({", ".join(f"{k}={v}" for k, v in args.items())})"
            ) from e

        if isinstance(result, ConfigVariable):
            return result.get_values(config, var_name)
        else:
            return [result]


def computed[T](
    compute: Callable[..., T | ConfigVariable[T]],
    dependencies: Iterable[Dependency] | None = None,
) -> ComputedConfigVariable[T]:
    return ComputedConfigVariable(compute, dependencies)


def conditional[T](
    condition: Callable[..., bool],
    then: T | ConfigVariable[T],
    else_: T | ConfigVariable[T],
) -> ComputedConfigVariable[T]:
    def compute(**kwargs):
        if condition(**kwargs):
            return then
        else:
            return else_

    return computed(compute, dependencies=infer_dependencies(condition))


def as_config_variable[T](value: T | ConfigVariable[T]) -> ConfigVariable[T]:
    if isinstance(value, ConfigVariable):
        return value
    else:
        return fixed(value)


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


class Config:
    _variables: ClassVar[dict[str, ConfigVariable]]

    def __init__(self, values: dict[str, Any], _i_know_what_i_am_doing=False):
        if type(self) is Config:
            raise TypeError("Cannot instantiate Config directly, use a subclass")

        if not _i_know_what_i_am_doing:
            raise TypeError(
                f"Cannot instantiate {type(self).__name__} directly, "
                f"use {type(self).__name__}.collect() instead"
            )

        self._values = values

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __repr__(self):
        return (
            f"{type(self).__name__}"
            f"({', '.join(f'{k}={v}' for k, v in self._values.items())})"
        )

    def __init_subclass__(cls):
        cls._variables = {
            name: as_config_variable(value)
            for name, value in cls.__dict__.items()
            if not name.startswith("_")
        }

    @classmethod
    def collect(cls) -> "ConfigCollection[Self]":
        dicts = [{}]

        # Argument is a single config -> expand ConfigVariable values
        # constants: dict[str, Any] = {}
        variables = cls._variables

        # configs.append(constants)

        for key, var in variables.items():
            available_dicts = dicts.copy()

            # print(f"Computing {key} with {len(available_configs)} configs")

            for dict_ in available_dicts:
                values = var.get_values(dict_, key)
                if len(values) == 0:
                    continue  # No values possible -> skip

                # Add first value to current config
                dict_[key] = values[0]
                # Create new configs for all other values
                for value in values[1:]:
                    dicts.append(dict_ | {key: value})

        return ConfigCollection(
            cls(dict_, _i_know_what_i_am_doing=True) for dict_ in dicts
        )

    def to_dict(self) -> dict[str, Any]:
        return self._values.copy()


class ConfigCollection[C: Config](Iterable[C]):
    def __init__(self, configs: Iterable[C]):
        self.configs = list(configs)

    def __getitem__(self, index: int) -> C:
        return self.configs[index]

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def __repr__(self):
        return (
            f"ConfigCollection[{len(self)}](\n  "
            + "\n".join(map(str, self.configs)).replace("\n", "\n  ")
            + "\n)"
        )

    def to_df(self):
        import pandas as pd

        return pd.DataFrame([config.to_dict() for config in self.configs])

    def run[T](self, func: Callable[[C], T]) -> list[tuple[C, T]]:
        return [(config, func(config)) for config in self.configs]


import numpy as np


class ExampleConfig(Config):
    a = choice(np.pi, 2 * np.pi)
    b = choice(30, 60)
    c = computed(lambda a, b: a * b)
    d = computed(lambda a, c: choice(a + c, a - c))
    e = conditional(lambda b: b > 40, 100, 200)


print(ExampleConfig.collect().to_df())
