from datetime import datetime
import os
from pathlib import Path
import re
import traceback
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:
    from pydantic import BaseModel


def _get_last_experiment_id():
    items = os.listdir(Path(__file__).parent)
    last_id = 0

    for item in items:
        if (match := re.match(r"E(\d+)_.*", item)) is not None:
            id = int(match.group(1))
            if id > last_id:
                last_id = id

    return last_id


def _format_id(id: int, *, digits: int = 3):
    s = str(id).rjust(digits, "0")

    if len(s) > digits:
        raise ValueError(f"ID overflow. IDs can only have up to {digits} digits")

    return s


class Experiment[C: BaseModel]:
    def __init__(
        self, path: Path, name: str, config_model: type[C], run_func: Callable[[C], Any]
    ):
        self.name = name
        self.config_model = config_model
        self.run_func = run_func
        self.path = path.relative_to(Path.cwd())

    def run(self, config: C, name: str | None = None):
        id_str = _format_id(_get_last_experiment_id() + 1)

        if name is not None:
            id_str += f"_{name}"

        print(f"{self.name} experiment #{id_str}: {config}")
        path = self.path / f"E{id_str}"

        if path.exists():
            raise ValueError(
                f"Path for {self.name} experiment #{id_str} already exists"
            )

        print("Running...")

        try:
            result = self.run_func(config)
        except Exception as e:
            print(f"Experiment failed: {e}")
            self._save_exception(config, id_str, traceback.format_exc())
        else:
            print("Experiment finished successfully")
            self._save_result(config, id_str, result)

    def _get_result_path(self, id_str: str):
        return self.path / f"E{id_str}" / "result.pickle"

    def _save_result(self, config: C, id_str: str, result: Any):
        import pickle, yaml, pprint

        result_path = self._get_result_path(id_str)
        result_path.parent.mkdir(parents=True)

        with result_path.open("wb") as result_file:
            pickle.dump(result, result_file)

        config_path = self.path / f"E{id_str}" / "config.yaml"

        with config_path.open("w") as config_file:
            yaml.dump(config.model_dump(), config_file)

        repr_path = self.path / f"E{id_str}" / "result.txt"
        repr_path.write_text(pprint.pformat(result))

        print(f'Result saved to "{result_path}"')

    def _load_result(self, id_str: str):
        import pickle

        result_path = self._get_result_path(id_str)

        with result_path.open("rb") as result_file:
            result = pickle.load(result_file)

        return result

    def _save_exception(self, config: C, id_str: str, formatted_exc: str):

        error_path = (
            self.path
            / "failed_experiments"
            / f"{id_str}_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        )
        error_path.parent.mkdir(exist_ok=True, parents=True)

        import yaml

        config_str = yaml.dump(config.model_dump())

        content = f"CONFIG:\n{config_str}\n\nEXCEPTION:\n{formatted_exc}"

        error_path.write_text(content)

        print(f"Exception details were written to {error_path}")

    def init_cli(self):
        import click

        @click.group()
        def cli():
            pass

        @cli.command()
        @click.option("--name", "-n")
        @click.option("--config", "-c", type=(str, str), multiple=True)
        def run(name, config):
            config = self.config_model(**dict(config))
            self.run(config, name)

        @cli.command()
        @click.argument("id_str")
        @click.option("--key", "-k")
        def show(id_str, key):
            result = self._load_result(id_str)

            if key is not None:
                result = result[key]

            print(repr(result))

        @cli.command()
        @click.argument("id_str")
        def keys(id_str):
            result = self._load_result(id_str)
            if not isinstance(result, Mapping):
                raise ValueError("Result is not a mapping")
            print("\n".join(result.keys()))

        cli()
