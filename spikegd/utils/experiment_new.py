from __future__ import annotations

import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import click
import yaml


class InvalidRunNameException(Exception):
    pass


class RunIdOverflowException(Exception):
    pass


class ExperimentRun:
    def __init__(
        self, experiment: ExperimentDefinition, id: int, suffix: str | None = None
    ):
        self.experiment = experiment
        self.id = id
        self.suffix = suffix

        self.name = str(id).rjust(self.experiment.id_length, "0")

        if len(self.name) > self.experiment.id_length:
            raise RunIdOverflowException(
                f"Run id overflow: {self.name}. Max length: {self.experiment.id_length}"
            )

        if self.suffix is not None:
            self.name += f"_{self.suffix}"

    @staticmethod
    def from_name(experiment: ExperimentDefinition, name: str):
        match = re.match(r"^(\d+)(?:_(.*))?$", name)

        if not match:
            raise InvalidRunNameException(f"Invalid run name: {name}")

        id = int(match.group(1))
        suffix = match.group(2)

        return ExperimentRun(experiment, id, suffix)

    def get_full_name(self):
        return f"{self.experiment.name}.{self.name}"

    def get_path(self):
        return self.experiment.runs_path / self.name

    def get_info(self):
        path = self.get_path() / "info.yaml"
        return yaml.safe_load(path.read_text())

    def set_info(self, **info):
        path = self.get_path() / "info.yaml"
        path.write_text(yaml.dump(info))

    def get_result(self):
        path = self.get_path() / "result.pickle"

        if not path.exists():
            return None

        with path.open("rb") as f:
            return pickle.load(f)

    def set_result(self, result: Any):
        path = self.get_path() / "result.pickle"
        with path.open("wb") as f:
            pickle.dump(result, f)

    def get_log(self):
        path = self.get_path() / "log.txt"

        if not path.exists():
            return ""

        return path.read_text()

    def set_log(self, log: str):
        path = self.get_path() / "log.txt"
        path.write_text(log)

    def log(self, log: str):
        path = self.get_path() / "log.txt"
        with path.open("a") as f:
            f.write(log)

    def update_info(self, **info):
        current_info = self.get_info()
        current_info.update(info)
        self.set_info(**current_info)

    def init(self):
        path = self.get_path()

        if path.exists():
            raise ValueError(f"Run already exists: {self}")

        path.mkdir(parents=True)

        shutil.copytree(self.experiment.src_path, path / "src")
        initialized_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.set_info(
            name=self.name,
            id=self.id,
            suffix=self.suffix,
            initialized_at=initialized_at,
        )

        return path

    def clear(self):
        path = self.get_path()
        shutil.rmtree(path)

    def __str__(self):
        return self.get_full_name()


class ExperimentDefinition:
    def __init__(self, main_path: str | Path, name: str | None = None, *, id_length=3):
        self.main_path = Path(main_path).relative_to(Path.cwd())
        if self.main_path.suffix != ".py":
            raise ValueError("Main path must be a .py file")

        self.root_path = self.main_path.parent
        self.src_path = self.root_path / "src"
        self.runs_path = self.root_path / "runs"

        self.name = name or self.root_path.name
        self.id_length = id_length

    def get_runs_dict(self) -> dict[int, ExperimentRun]:
        if not self.runs_path.exists():
            return {}

        runs_dict: dict[int, ExperimentRun] = {}

        for run_path in self.runs_path.iterdir():
            try:
                run = ExperimentRun.from_name(self, run_path.name)

                if run.id in runs_dict:
                    raise ValueError(f"Duplicate run id: {run.id}")

                runs_dict[run.id] = run
            except InvalidRunNameException:
                # print(f"Skipping invalid run name: {run_path.name}")
                pass

        return runs_dict

    def get_runs(self):
        runs = list(self.get_runs_dict().values())
        runs.sort(key=lambda run: run.id)
        return runs

    def get_run_names(self):
        return [run.name for run in self.get_runs()]

    def get_run_ids(self) -> list[int]:
        ids = list(self.get_runs_dict().keys())
        ids.sort()
        return ids

    def get_last_run_id(self):
        ids = self.get_run_ids()
        return ids[-1] if ids else None

    def get_next_run_id(self):
        last_id = self.get_last_run_id()
        return last_id + 1 if last_id is not None else 1

    def get_next_run(self, suffix: str | None = None):
        return ExperimentRun(self, self.get_next_run_id(), suffix)

    def get_run(
        self, id: int | str, *, suffix: str | None = None, allow_next=False
    ) -> ExperimentRun:
        if id == "last":
            runs = self.get_runs()
            if not runs:
                raise ValueError("No runs found")
            return runs[-1]
        elif id == "next" and allow_next:
            return self.get_next_run(suffix)
        else:
            run = self.get_runs_dict().get(int(id))
            if run is None:
                raise ValueError(f"No run found with id {id}")
            return run

    def get_run_or_none(
        self, id: int | Literal["last", "next"], *, suffix: str | None = None
    ) -> ExperimentRun | None:
        try:
            return self.get_run(id, suffix=suffix)
        except ValueError:
            return None

    def init_cli(self):
        @click.group()
        def cli():
            pass

        @cli.command()
        def runs():
            names = self.get_run_names()

            if not names:
                click.echo("No runs found")
            else:
                click.echo(f"{len(names)} runs found:")
                click.echo("\n".join(names))

        @cli.command()
        @click.argument("id")
        @click.option("--suffix", "-s", default=None)
        def init(id, suffix):
            run = self.get_run(id, suffix=suffix, allow_next=True)
            path = run.init()
            click.echo(f"Initialized run {run} at {path}")

        @cli.command()
        @click.argument("id")
        def result(id):
            run = self.get_run(id)
            result = run.get_result()

            if result is None:
                click.echo(f"No result found for run {run}")
            else:
                click.echo(result)

        @cli.command()
        @click.argument("id")
        def log(id):
            run = self.get_run(id)
            log = run.get_log()

            if not log:
                click.echo("<empty>")
            else:
                click.echo(log)

        cli()
