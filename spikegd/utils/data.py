from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import pandas as pd

most_frequent = object()


class Data:
    _df: pd.DataFrame
    _inputs: tuple[str, ...]
    _input_consts: dict[str, Any]
    _input_vars: dict[str, dict[Any, int]]
    _outputs: set[str]
    _history: list[str]
    _outer_data: Data | None

    def __init__(self, df: pd.DataFrame, inputs: Iterable[str]):
        self._df = df.copy()
        self._inputs = tuple(inputs)
        self._history = []
        self._outer_data = None
        self._compute_metadata()

    @property
    def df(self):
        return self._df

    @property
    def inputs(self):
        return self._inputs

    @property
    def input_consts(self) -> Mapping[str, Any]:
        return self._input_consts

    @property
    def input_vars(self) -> Mapping[str, Mapping[Any, int]]:
        return self._input_vars

    @property
    def input_var_df(self) -> pd.DataFrame:
        return self.df[list(self.input_vars)]

    @property
    def outputs(self) -> Iterable[str]:
        return self._outputs

    @property
    def output_df(self) -> pd.DataFrame:
        return self.df[list(self.outputs)]

    @property
    def history(self) -> Iterable[str]:
        return self._history

    # @indep_cols.setter
    # def indep_cols(self, value: Iterable[str]):
    #     self._indep_cols = tuple(value)
    #     self._compute_metadata()

    def _compute_metadata(self):
        self._input_consts = {}
        self._input_vars = {}
        self._outputs = set()

        for col in self._inputs:
            if col not in self._df.columns:
                raise ValueError(f"Independent column {col!r} not in DataFrame")

            value_counts = self._df[col].value_counts(dropna=False)

            if len(value_counts) == 1:
                self._input_consts[col] = value_counts.index[0]
            else:
                self._input_vars[col] = value_counts.to_dict()

        for col in self._df.columns:
            if col not in self._inputs:
                self._outputs.add(col)

    def update(self, new_df: pd.DataFrame, name: str):
        self._df = new_df
        self._history.append(name)
        self._compute_metadata()

        return self

    def transform(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame | None],
        name: str | None = None,
    ):
        result = func(self._df)

        if isinstance(result, pd.DataFrame):
            self._df = result
        elif result is not None:
            raise ValueError(f"Invalid return value from transformation: {result}")

        self._history.append(name or func.__name__)
        self._compute_metadata()

        return self

    def computed(self, col: str, func: Callable[[pd.DataFrame], Any] | str):
        if isinstance(func, str):
            func_str = func
            func = lambda df: df.eval(func_str)  # noqa: E731

        return self.update(
            self.df.assign(**{col: func(self.df)}),
            f"computed: {col} = {func.__name__}",
        )

    def _fix_one(self, col: str, value: Any = most_frequent):
        if col not in self._inputs:
            raise ValueError(f"Cannot fix non-input column {col!r}")

        if col not in self._input_vars:
            return

        if value is most_frequent:
            value_counts = self._input_vars[col]
            value = max(value_counts, key=value_counts.__getitem__)

        self.update(self._df[self._df[col] == value], f"fix: {col} = {value!r}")

    def fix(
        self,
        key_or_values: str | Mapping[str, Any] | None = None,
        value: Any = most_frequent,
        /,
        **values,
    ):
        if isinstance(key_or_values, str):
            self._fix_one(key_or_values, value)
        elif isinstance(key_or_values, Mapping):
            for col, value in key_or_values.items():
                self._fix_one(col, value)
        elif key_or_values is not None:
            raise ValueError(f"Invalid first arg: {key_or_values}")

        for col, value in values.items():
            self._fix_one(col, value)

        return self

    def fix_all(self, *include: str, exclude: Iterable[str] = ()):
        for key in include or self.inputs:
            if key in exclude or key not in self._input_vars:
                continue

            self._fix_one(key, most_frequent)

        return self

    def apply_value_whitelist(self, key: str, values: Iterable[Any]):
        return self.update(
            self.df[self.df[key].isin(values)],
            f"apply_value_whitelist: {key} in {values}",
        )

    def apply_value_blacklist(self, key: str, values: Iterable[Any]):
        return self.update(
            self.df[~self.df[key].isin(values)],
            f"apply_value_blacklist: {key} not in {values}",
        )

    def apply_limits(self, key: str, min_: Any | None = None, max_: Any | None = None):
        if min_ is not None:
            self.update(
                self.df[self.df[key] >= min_],
                f"apply_limits: {key} >= {min_!r}",
            )

        if max_ is not None:
            self.update(
                self.df[self.df[key] <= max_],
                f"apply_limits: {key} <= {max_!r}",
            )

        return self

    def aggregate(self, agg_func: Any, group_col: str):
        return self.update(
            self.df.groupby(group_col).aggregate(agg_func).reset_index(),
            f"aggregate: {agg_func} for equal {group_col}",
        )

    def drop(self, *cols: str):
        return self.update(self.df.drop(columns=cols), f"drop: {', '.join(cols)}")

    def schema(self, include_funcs: bool = False):
        s = "Data:\n"

        for col, value in self.input_consts.items():
            s += f"  const {col} = {value!r}\n"

        for col, value_counts in self.input_vars.items():
            values_str = ", ".join(
                f"{value!r} ({count}x)" for value, count in value_counts.items()
            )
            s += f"  var {col} = {values_str}\n"

        if include_funcs:
            for func in self.outputs:
                s += f"  func {func}\n"

        s = s.rstrip()

        if "\n" not in s:
            s += " (empty)"

        return s

    def values(self, col: str):
        return self._df[col].unique()

    def __repr__(self):
        return repr(self._df)

    def copy(self):
        data = Data(self._df, self._inputs)
        data._history = self._history.copy()
        return data

    def _set_state(self, state: Data):
        self._df = state._df
        self._inputs = state._inputs
        self._input_consts = state._input_consts
        self._input_vars = state._input_vars
        self._outputs = state._outputs
        self._history = state._history

    def __enter__(self):
        self._outer_data = self.copy()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._outer_data is None:
            raise ValueError("Data.__exit__ called without Data.__enter__")

        self._set_state(self._outer_data)
        self._outer_data = None
        return False
