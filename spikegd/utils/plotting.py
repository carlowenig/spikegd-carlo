"""Figure options and helper functions for plotting."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .formatting import fmt, fmt_count, fmt_dict, fmt_list


def cm2inch(x: float, y: float) -> tuple[float, float]:
    """Convert cm to inch."""
    inch = 2.54
    return x / inch, y / inch


def panel_label(
    fig: Figure, ax: Axes, label: str, x: float = 0.0, y: float = 0.0
) -> None:
    """Add panel label to figure."""
    trans = mtransforms.ScaledTranslation(x, y, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="large",
    )


# Formatter for log axis ticks
formatter = mticker.FuncFormatter(lambda y, _: "{:.16g}".format(y))
loc_major = mticker.LogLocator(
    base=10.0,
    subs=(
        0.1,
        1.0,
    ),
    numticks=12,
)
loc_min = mticker.LogLocator(
    base=10.0, subs=tuple(jnp.arange(0.1, 1.0, 0.1)), numticks=12
)

# Colorblind-friendly colors from https://arxiv.org/abs/2107.02270,
# see also https://github.com/matplotlib/matplotlib/issues/9460.
petroff10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

# Color maps
cmap_grays = plt.get_cmap("gray_r")
cmap_blues = plt.get_cmap("Blues")
cmap_oranges = plt.get_cmap("Oranges")
cmap_purples = plt.get_cmap("Purples")


# Plot dataframe
def _get_val_list(df, key, min_count=0) -> list:
    if not key:
        return []

    val_counts = df[key].value_counts()
    return val_counts[val_counts >= min_count].index.sort_values().to_list()


def _expand_keys(
    keys: str | tuple[str, ...] | None,
    available_keys: tuple[str, ...],
    raise_if_unavailable=True,
) -> tuple[str, ...]:
    """Expand zero or more keys to a tuple of keys. Allows wildcard at start or end."""
    if keys is None:
        return ()

    if isinstance(keys, str):
        keys = (keys,)

    expanded_keys = []

    for key in keys:
        # Globbing
        if key[0] == "*":
            if "*" in key[1:]:
                raise ValueError("Only one wildcard allowed")

            expanded_keys.extend(k for k in available_keys if k.endswith(key[1:]))
        elif key[-1] == "*":
            if "*" in key[:-1]:
                raise ValueError("Only one wildcard allowed")

            expanded_keys.extend(k for k in available_keys if k.startswith(key[:-1]))
        elif "*" in key:
            raise ValueError("Wildcard only allowed at start or end")
        elif key not in available_keys:
            if raise_if_unavailable:
                raise ValueError(f"Key {key} not in available keys {available_keys}")
            else:
                continue
        else:
            expanded_keys.append(key)

    # Return unique keys
    return tuple(dict.fromkeys(expanded_keys))


class PlotType(ABC):
    @abstractmethod
    def draw_to_ax(
        self,
        grid: "PlotGrid",
        full_df: pd.DataFrame,
        ax: Axes,
        ax_df: pd.DataFrame,
        func_key: str,
    ) -> None: ...

    def style_grid(
        self, grid: "PlotGrid", fig: Figure, axs: np.ndarray, full_df: pd.DataFrame
    ) -> None:
        pass

    def get_used_keys(self) -> list[str]:
        return []

    def get_preferred_size(self, n_rows: int, n_cols: int) -> tuple[float, float]:
        width = 7 if n_cols == 1 else 3.5
        height = 5 if n_rows == 1 else 2
        return width, height


@dataclass
class LinePlot(PlotType):
    graph_key: str | None = None
    graph_cmap: str = "tab10"
    x_scale: str = "linear"
    y_scale: str = "linear"

    def get_used_keys(self):
        return [self.graph_key] if self.graph_key else []

    def draw_to_ax(
        self,
        grid: "PlotGrid",
        full_df: pd.DataFrame,
        ax: Axes,
        ax_df: pd.DataFrame,
        func_key: str,
    ):
        if len(grid.arg_keys) != 1:
            raise ValueError(
                "LinePlot requires exactly one arg key since it is 1-dimensional"
            )

        arg_key = grid.arg_keys[0]

        graph_cmap = plt.get_cmap(self.graph_cmap)
        graph_vals = _get_val_list(ax_df, self.graph_key) or [None]

        y_count_vals = ax_df.groupby(arg_key).count()[func_key].unique()
        all_y_counts_equal = len(y_count_vals) == 1

        # Display counts
        if all_y_counts_equal:
            ax.text(
                1,
                1.02,
                f"{fmt_count(y_count_vals[0], "sample")} per dot",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="right",
            )

        for graph_index, graph_val in enumerate(graph_vals):
            graph_df = ax_df

            if graph_val is not None:
                graph_df = graph_df[graph_df[self.graph_key] == graph_val]

            grouped = graph_df.groupby(arg_key)
            mean_df = grouped.mean()
            std_df = grouped.std()
            count_df = grouped.count()
            # print(count_df)

            arg_arr = mean_df.index
            func_mean = mean_df[func_key]
            func_std = std_df[func_key]
            func_count = count_df[func_key]
            func_mean_err = func_std / np.sqrt(func_count)

            color = graph_cmap(graph_index)

            multiple_per_dot = np.any(func_count > 1)
            # all_counts_equal = y_count.nunique() == 1

            # Plot mean
            ax.plot(
                arg_arr,
                func_mean,
                marker="o",
                markersize=8 if multiple_per_dot and not all_y_counts_equal else 4,
                linestyle="--",
                color=color,
                zorder=graph_index,
                label=fmt(graph_val) if self.graph_key else None,
            )

            if multiple_per_dot and not all_y_counts_equal:
                for x_val in arg_arr:
                    ax.text(
                        x_val,
                        func_mean[x_val],  # type: ignore
                        func_count[x_val],  # type: ignore
                        fontsize=6,
                        fontweight="bold",
                        va="center_baseline",
                        ha="center",
                        color="white",
                        zorder=graph_index,
                    )

            # Plot 1-sigma error
            ax.fill_between(
                arg_arr,
                func_mean - func_mean_err,
                func_mean + func_mean_err,
                alpha=0.15,
                color=color,
                zorder=graph_index - 1000,
            )

            # Plot all samples
            if multiple_per_dot:
                ax.plot(
                    graph_df[arg_key],
                    graph_df[func_key],
                    marker="x",
                    markersize=3,
                    linestyle="",
                    color=color,
                    alpha=0.5,
                    zorder=-2000,
                )

        ax.grid(alpha=0.4, which="major")
        ax.grid(alpha=0.2, which="minor")
        ax.set_xscale(self.x_scale)
        ax.set_yscale(self.y_scale)

    def style_grid(
        self, grid: "PlotGrid", fig: Figure, axs: np.ndarray, full_df: pd.DataFrame
    ):
        # Add x-label to last row of each column
        for ax in axs[-1]:
            ax.set_xlabel(fmt(grid.arg_keys[0], grid.key_format))

        # Add y-label to first column of each row
        n_rows_per_func = axs.shape[0] // len(grid.func_keys)
        for i, func_key in enumerate(grid.func_keys):
            for ax in axs[i::n_rows_per_func, 0]:
                ax.set_ylabel(fmt(func_key, grid.key_format))

        if self.graph_key:
            axs[-1, -1].legend(
                loc="lower left",
                bbox_to_anchor=(1, 0),
                title=fmt(self.graph_key),
            )


def _set_value_ticks(ax, axis, vals, max_ticks=8):
    step = 1
    indices = np.arange(0, len(vals))
    while len(indices) > max_ticks:
        indices = np.arange(0, len(vals), step)
        step += 1

    labels = np.asarray(vals)[indices]

    if axis == "x":
        ax.set_xticks(indices)
        ax.set_xticklabels(labels)
    elif axis == "y":
        ax.set_yticks(indices)
        ax.set_yticklabels(labels)
    else:
        raise ValueError(f"Invalid axis {axis!r}")


@dataclass
class HeatmapPlot(PlotType):
    # color_key: str
    cmap: str = "viridis"
    norm: type = mcolors.Normalize

    # def get_used_keys(self):
    #     return [self.color_key]

    def _get_func_limits(self, grid: "PlotGrid", full_df, func_key):
        min, max = grid.limits.get(func_key, (None, None))

        if min is None:
            min = full_df[func_key].min()

        if max is None:
            max = full_df[func_key].max()

        return min, max

    def draw_to_ax(
        self,
        grid: "PlotGrid",
        full_df: pd.DataFrame,
        ax: Axes,
        ax_df: pd.DataFrame,
        func_key: str,
    ):
        if len(grid.arg_keys) != 2:
            raise ValueError(
                "HeatmapPlot requires exactly two arg keys since it is 2-dimensional"
            )

        x_key, y_key = grid.arg_keys

        x_vals = _get_val_list(ax_df, x_key)
        y_vals = _get_val_list(ax_df, y_key)

        if not x_vals or not x_vals:
            return

        x_vals = np.sort(x_vals)
        y_vals = np.sort(y_vals)

        data = np.zeros((len(x_vals), len(y_vals)))

        for x_i, x_val in enumerate(x_vals):
            for y_i, y_val in enumerate(y_vals):
                mask = (ax_df[x_key] == x_val) & (ax_df[y_key] == y_val)
                data[x_i, y_i] = ax_df[mask][func_key].mean()

        # Use full_df to make sure vmin and vmax are consistent across all subplots
        func_min, func_max = self._get_func_limits(grid, full_df, func_key)

        ax.imshow(
            data.T,
            cmap=self.cmap,
            aspect="auto",
            norm=self.norm(vmin=func_min, vmax=func_max),
            interpolation="nearest",
            origin="lower",
        )

        _set_value_ticks(ax, "x", x_vals)
        _set_value_ticks(ax, "y", y_vals)

    def style_grid(
        self, grid: "PlotGrid", fig: Figure, axs: np.ndarray, full_df: pd.DataFrame
    ):
        x_key, y_key = grid.arg_keys

        # Add x-label to last row of each column
        for ax in axs[-1]:
            ax.set_xlabel(fmt(x_key, grid.key_format))

        # Add y-label to first column of each row
        for ax in axs[:, 0]:
            ax.set_ylabel(fmt(y_key, grid.key_format))

        n_rows_per_func = axs.shape[0] // len(grid.func_keys)
        for i, func_key in enumerate(grid.func_keys):
            # cax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
            func_min, func_max = self._get_func_limits(grid, full_df, func_key)

            mappable = plt.cm.ScalarMappable(
                norm=self.norm(vmin=func_min, vmax=func_max), cmap=self.cmap
            )
            label = fmt(func_key, grid.key_format)

            # Add colorbar to last column of each row corresponding to func_key
            for ax in axs[i::n_rows_per_func, -1]:
                fig.colorbar(mappable, ax=ax, label=label)

    def get_preferred_size(self, n_rows: int, n_cols: int):
        if n_rows == n_cols == 1:
            return 5, 4
        else:
            return 3, 3.5


Limits = tuple[float | None, float | None]


@dataclass
class PlotGrid:
    arg_keys: tuple[str, ...]
    func_keys: tuple[str, ...]
    plot_type: PlotType | None = None
    col_key: str | None = None
    row_key: str | None = None
    fixed_values: dict[str, Any] = field(default_factory=dict)
    mean_keys: tuple[str, ...] | str | None = None
    indep_keys: tuple[str, ...] | str | None = None
    min_points_per_val = 0
    key_format: Any = None
    limits: dict[str, Limits] = field(default_factory=dict)
    whitelists: dict[str, list] = field(default_factory=dict)

    def get_plot_type(self) -> PlotType:
        if self.plot_type is not None:
            return self.plot_type
        elif len(self.arg_keys) == 1:
            return LinePlot()
        elif len(self.arg_keys) == 2:
            return HeatmapPlot()
        else:
            raise ValueError("No plot object specified and cannot infer plot type")

    def show(self, df):
        total_samples = len(df)
        plot_type = self.get_plot_type()

        # Filter fixed values
        for fixed_key, fixed_value in self.fixed_values.items():
            df = df[df[fixed_key] == fixed_value]

        # Only keep numeric columns
        df = df.select_dtypes(include="number")

        available_keys = tuple(df.columns)
        func_keys = _expand_keys(self.func_keys, available_keys)
        n_funcs = len(func_keys)
        mean_keys = _expand_keys(self.mean_keys, available_keys)
        fixed_values = self.fixed_values.copy()

        # Apply whitelists
        for whitelist_key, whitelist in self.whitelists.items():
            df = df[df[whitelist_key].isin(whitelist)]

        # Apply limits
        for limit_key, (min_val, max_val) in self.limits.items():
            if min_val is not None:
                df = df[df[limit_key] >= min_val]
            if max_val is not None:
                df = df[df[limit_key] <= max_val]

        # Check if independent keys are constant
        if self.indep_keys is not None:
            used_keys = {
                *self.arg_keys,
                *func_keys,
                self.col_key,
                self.row_key,
                *mean_keys,
                *plot_type.get_used_keys(),
            }

            indep_keys = _expand_keys(
                self.indep_keys, available_keys, raise_if_unavailable=False
            )
            for key in indep_keys:
                if key in used_keys:
                    continue

                # Independent key not used -> check if it is constant
                if len(val_counts := df[key].value_counts()) > 1:
                    most_freq_val = val_counts.idxmax()
                    print(
                        f"Independent key {key!r} has multiple values: {val_counts.index.tolist()}. "
                        f"Using most frequent value: {most_freq_val}"
                    )
                    df = df[df[key] == most_freq_val]
                    fixed_values[key] = most_freq_val

        col_vals = _get_val_list(df, self.col_key, self.min_points_per_val) or [None]
        n_cols = len(col_vals)

        row_vals = _get_val_list(df, self.row_key, self.min_points_per_val) or [None]
        n_rows = len(row_vals)
        n_rows_total = n_rows * n_funcs

        width_per_plot, height_per_plot = plot_type.get_preferred_size(
            n_rows_total, n_cols
        )

        width = width_per_plot * n_cols
        height = height_per_plot * n_rows_total

        fig, axs = plt.subplots(
            n_rows_total,
            n_cols,
            figsize=(width, height),
            dpi=200,
            sharey="row",
            sharex="col",
            squeeze=False,
            layout="compressed",
        )
        # graph_cmap = plt.get_cmap(graph_cmap)

        # print(f"Creating {axs.shape[0]}x{axs.shape[1]} subplots")

        for row_i, col_i, func_i in np.ndindex(n_rows, n_cols, n_funcs):
            row_val = row_vals[row_i]
            col_val = col_vals[col_i]
            func_key = func_keys[func_i]
            ax = axs[row_i * n_funcs + func_i, col_i]

            plot_vals = {}
            if self.row_key:
                plot_vals[self.row_key] = row_val
            if self.col_key:
                plot_vals[self.col_key] = col_val

            ax_df = df.copy()
            for key, value in plot_vals.items():
                ax_df = ax_df[ax_df[key] == value]

            plot_type.draw_to_ax(self, df, ax, ax_df, func_key)

            if plot_vals:
                ax.text(
                    0,
                    1.02,
                    fmt_dict(plot_vals, key_format=self.key_format),
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="left",
                )

        plot_type.style_grid(self, fig, axs, df)

        plt.suptitle(
            (f"Mean over {fmt_list(mean_keys, self.key_format)}\n" if mean_keys else "")
            + (
                f"with {fmt_dict(fixed_values, key_format=self.key_format)}\n"
                if fixed_values
                else ""
            )
            + f"(contains {len(df)} of {total_samples} samples)"
        )
        # plt.tight_layout()
        plt.show()
