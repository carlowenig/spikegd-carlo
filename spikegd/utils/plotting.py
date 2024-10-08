"""Figure options and helper functions for plotting."""

import itertools
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

import attrs
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .attrs_utils import UniqueCollector, as_converter
from .formatting import fmt, fmt_dict, fmt_intersection, fmt_list, fmt_number


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
def _get_val_list(df, key, min_count=0, verbose=False) -> list:
    if not key or key not in df.columns:
        return []

    val_counts = df[key].value_counts()

    mask = val_counts >= min_count

    if verbose and not mask.all():
        print(
            f"Ignoring values of {key!r} since they have less than {min_count} entries: {val_counts[~mask].index.sort_values().tolist()}"
        )

    return val_counts[mask].index.sort_values().to_list()


# def _expand_keys(
#     keys: str | tuple[str, ...] | None,
#     available_keys: tuple[str, ...],
#     raise_if_unavailable=True,
# ) -> tuple[str, ...]:
#     """Expand zero or more keys to a tuple of keys. Allows wildcard at start or end."""
#     if keys is None:
#         return ()

#     if isinstance(keys, str):
#         keys = (keys,)

#     expanded_keys = []

#     for key in keys:
#         # Globbing
#         if key[0] == "*":
#             if "*" in key[1:]:
#                 raise ValueError("Only one wildcard allowed")

#             expanded_keys.extend(k for k in available_keys if k.endswith(key[1:]))
#         elif key[-1] == "*":
#             if "*" in key[:-1]:
#                 raise ValueError("Only one wildcard allowed")

#             expanded_keys.extend(k for k in available_keys if k.startswith(key[:-1]))
#         elif "*" in key:
#             raise ValueError("Wildcard only allowed at start or end")
#         elif key not in available_keys:
#             if raise_if_unavailable:
#                 raise ValueError(f"Key {key} not in available keys {available_keys}")
#             else:
#                 continue
#         else:
#             expanded_keys.append(key)

#     # Return unique keys
#     return tuple(dict.fromkeys(expanded_keys))


class Plot(ABC):
    @abstractmethod
    def draw_to_ax(
        self,
        grid: "PlotGrid",
        full_df: pd.DataFrame,
        ax: Axes,
        ax_df: pd.DataFrame,
        func_key: str,
        func_agg: str | None,
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


@attrs.define(frozen=True)
class LinePlot(Plot):
    graph_key: str | None = attrs.field(default=None)
    graph_cmap: mcolors.Colormap = attrs.field(
        default=plt.get_cmap("tab10"), converter=plt.get_cmap
    )
    graph_cmap_range: Literal["auto", "full"] | tuple[float, float] = attrs.field(
        default="auto"
    )
    x_scale: str = attrs.field(default="linear")
    y_scale: str = attrs.field(default="linear")
    corridor: Literal["std", "err"] | None = attrs.field(
        default=None, validator=attrs.validators.in_((None, "std", "err"))
    )

    def get_used_keys(self):
        return [self.graph_key] if self.graph_key else []

    def draw_to_ax(
        self,
        grid: "PlotGrid",
        full_df: pd.DataFrame,
        ax: Axes,
        ax_df: pd.DataFrame,
        func_key: str,
        func_agg: str | None,
    ):
        if len(grid.arg_keys) != 1:
            raise ValueError(
                "LinePlot requires exactly one arg key since it is 1-dimensional"
            )

        arg_key = grid.arg_keys[0]

        graph_cmap = plt.get_cmap(self.graph_cmap)
        # get graph values from full_df to make sure that every ax uses the same colors,
        # such that a single legend can be used
        graph_vals = _get_val_list(full_df, self.graph_key) or [None]

        for graph_index, graph_val in enumerate(graph_vals):
            graph_df = ax_df

            if graph_val is not None:
                graph_df = graph_df[graph_df[self.graph_key] == graph_val]

            grouped = graph_df.groupby(arg_key)

            # count of valid func values per arg value
            func_counts = grouped[func_key].count()
            show_counts = np.any(func_counts > 1)

            arg_vals = np.array(list(grouped.groups.keys()))

            if func_agg is not None:
                # function mean over same arg values
                func_vals = grouped[func_key].agg(func_agg)
            elif np.all(func_counts == 1):
                func_vals = grouped[func_key].first()
            else:
                faulty_arg_vals = func_counts[func_counts != 1].index

                if grid.indep_keys is not None:
                    for arg_val in faulty_arg_vals:
                        group = grouped.get_group(arg_val)
                        for key in grid.indep_keys:
                            if key in group.columns and group[key].nunique() > 1:
                                if key in grid.agg_keys:
                                    raise ValueError(
                                        f"Cannot aggregate over multiple values varying in {key} since no func_agg is specified"
                                    )
                                else:
                                    raise ValueError(
                                        f"Independent key {key} differs for {arg_key} = {arg_val} (did you forget to specify it as an agg_key?)"
                                    )

                for arg_val in faulty_arg_vals:
                    group = grouped.get_group(arg_val)
                    print(
                        f"WARNING: Found multiple values for {arg_key} = {arg_val}. Using mean.\n"
                        f"  (config hashes: {fmt_list(group["config_hash"])})"
                    )

                func_vals = grouped[func_key].mean()

            assert len(arg_vals) == len(func_vals) == len(func_counts)

            normalized_graph_index = (
                graph_index / (len(graph_vals) - 1) if len(graph_vals) > 1 else 0
            )

            if self.graph_cmap_range == "auto":
                # use index modulo cmap resolution as integer index
                color = graph_cmap(graph_index % graph_cmap.N)
            elif self.graph_cmap_range == "full":
                # spread index to full cmap range
                color = graph_cmap(normalized_graph_index)
            elif (
                isinstance(self.graph_cmap_range, tuple)
                and len(self.graph_cmap_range) == 2
            ):
                cmap_start, cmap_end = self.graph_cmap_range
                color = graph_cmap(
                    cmap_start + normalized_graph_index / (cmap_end - cmap_start)
                )
            else:
                raise ValueError(
                    f"Invalid graph_cmap_range value {self.graph_cmap_range!r}"
                )

            # Plot mean
            ax.plot(
                arg_vals,
                func_vals,
                marker="o",
                markersize=10 if show_counts else 4,
                linestyle="--",
                color=color,
                zorder=graph_index,
                label=fmt(graph_val) if self.graph_key else None,
            )

            if show_counts:
                for arg_val in arg_vals:
                    ax.text(
                        arg_val,
                        func_vals[arg_val],  # type: ignore
                        func_counts[arg_val],  # type: ignore
                        fontsize=6,
                        fontweight="bold",
                        va="center_baseline",
                        ha="center",
                        color="white",
                        zorder=graph_index,
                    )

            if self.corridor is not None:
                # Plot 1-sigma error
                if self.corridor == "err":
                    if func_key.endswith("_mean"):
                        func_std_key = func_key.removesuffix("_mean") + "_std"

                        if func_std_key not in graph_df.columns:
                            raise ValueError(
                                f"Corridor='err' requires {func_std_key} to be present in the DataFrame"
                            )

                        corridor_width = (
                            grouped[func_std_key].agg(lambda x: np.sqrt(np.sum(x**2)))
                            / func_counts
                        )
                    else:
                        raise ValueError(
                            "Corridor='err' only supported for functions ending in '_mean'"
                        )
                elif self.corridor == "std":
                    corridor_width = grouped[func_key].std()
                else:
                    raise ValueError(f"Invalid corridor value {self.corridor!r}")

                corridor_width = corridor_width.fillna(0)

                ax.fill_between(
                    arg_vals,
                    func_vals - corridor_width,
                    func_vals + corridor_width,
                    alpha=0.15,
                    color=color,
                    zorder=graph_index - 1000,
                )

            # Plot all samples
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
            ylabel = grid.get_func_label(func_key)

            for ax in axs[i::n_rows_per_func, 0]:
                ax.set_ylabel(ylabel)

        if self.graph_key:
            axs[-1, -1].legend(
                loc="lower left",
                bbox_to_anchor=(1, 0),
                title=fmt(self.graph_key),
            )


type Axis = Literal["x", "y"]


def _set_value_ticks(ax: Axes, axis: Axis, vals, max_ticks: int | None = 8, **kwargs):
    step = 1
    indices = np.arange(0, len(vals))
    while max_ticks is not None and len(indices) > max_ticks:
        indices = np.arange(0, len(vals), step)
        step += 1

    labels = [fmt_number(val) for val in np.asarray(vals)[indices]]

    if axis == "x":
        ax.set_xticks(indices, labels, **kwargs)
    elif axis == "y":
        ax.set_yticks(indices, labels, **kwargs)
    else:
        raise ValueError(f"Invalid axis {axis!r}")


Color = tuple[float, float, float, float]


def get_luminance(color, background="white"):
    r, g, b, _ = overlay_colors(background, color)

    return r * 0.2126 + g * 0.7152 + b * 0.0722


def _overlay_2_colors(color1: Color, color2: Color) -> Color:
    *rgb1, a1 = color1
    *rgb2, a2 = color2

    a = a2 + a1 * (1 - a2)
    rgb = tuple((c2 * a2 + c1 * a1 * (1 - a2)) / a for c1, c2 in zip(rgb1, rgb2))

    return *rgb, a  # type: ignore


def overlay_colors(*colors) -> Color:
    if len(colors) == 0:
        raise ValueError("Need at least one color to overlay")

    result = mcolors.to_rgba(colors[0])
    for color in colors[1:]:
        result = _overlay_2_colors(result, mcolors.to_rgba(color))

    return result


def apply_alpha(color, alpha: float) -> Color:
    color = mcolors.to_rgba(color)
    return color[:3] + (color[3] * alpha,)


# 0% black on 100% white -> 100% white
assert overlay_colors("white", (0, 0, 0, 0)) == (1, 1, 1, 1)

# 50% black on 100% white -> 100% gray
assert overlay_colors("white", (0, 0, 0, 0.5)) == (0.5, 0.5, 0.5, 1)

# 50% white on 100% red -> 100% bright red
assert overlay_colors((1, 0, 0, 1), (1, 1, 1, 0.5)) == (1, 0.5, 0.5, 1)

# 50% black on 50% black background -> 75% black
assert overlay_colors((0, 0, 0, 0.5), (0, 0, 0, 0.5)) == (0, 0, 0, 0.75)

# 50% green on 100% blue -> 100% cyan
assert overlay_colors((0, 0, 1, 1), (0, 1, 0, 0.5)) == (0, 0.5, 0.5, 1)


def get_contrast_color(color: Color) -> Color:
    luminance = get_luminance(color)
    return (0, 0, 0, 1) if luminance > 0.5 else (1, 1, 1, 1)


@attrs.define(frozen=True)
class ValueTicks(Plot):
    arg_axes: Mapping[str, Axis] | Sequence[Axis] = attrs.field(default=("x", "y"))
    max_ticks: int | None = attrs.field(default=None)
    text_kwargs: dict[str, Any] = attrs.field(factory=dict)

    def draw_to_ax(
        self,
        grid: "PlotGrid",
        full_df: pd.DataFrame,
        ax: Axes,
        ax_df: pd.DataFrame,
        func_key: str,
        func_agg: str | None,
    ):
        if isinstance(self.arg_axes, Mapping):
            arg_axes: list[Axis | None] = [
                self.arg_axes.get(arg_key, None) for arg_key in grid.arg_keys
            ]
        else:
            arg_axes = list(self.arg_axes)

        for arg_axis, arg_key in zip(arg_axes, grid.arg_keys):
            if arg_axis is None:
                continue

            vals = _get_val_list(full_df, arg_key, grid.min_points_per_ax)
            _set_value_ticks(ax, arg_axis, vals, self.max_ticks, **self.text_kwargs)


@attrs.define(frozen=True)
class HeatmapPlot(Plot):
    cmap: mcolors.Colormap = attrs.field(
        default=plt.get_cmap("viridis"), converter=plt.get_cmap
    )
    norm: type = attrs.field(default=mcolors.Normalize)
    show_counts: Literal["never", ">1", "always"] = attrs.field(
        default=">1",
        validator=attrs.validators.in_(("never", ">1", "always")),
    )

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
        func_agg: str | None,
    ):
        if len(grid.arg_keys) != 2:
            raise ValueError(
                "HeatmapPlot requires exactly two arg keys since it is 2-dimensional"
            )

        x_key, y_key = grid.arg_keys

        x_vals = _get_val_list(full_df, x_key, grid.min_points_per_ax)
        y_vals = _get_val_list(full_df, y_key, grid.min_points_per_ax)

        if not x_vals or not y_vals:
            return

        x_vals = np.sort(x_vals)
        y_vals = np.sort(y_vals)
        shape = (len(x_vals), len(y_vals))

        data = np.zeros(shape)
        counts = np.zeros(shape, dtype=int)

        for x_i, y_i in np.ndindex(shape):
            mask = (ax_df[x_key] == x_vals[x_i]) & (ax_df[y_key] == y_vals[y_i])
            count = len(ax_df[mask])
            func_vals = ax_df[mask][func_key]

            if func_agg is not None:
                func_val = func_vals.agg(func_agg)
            elif count == 1:
                func_val = func_vals.iloc[0]
            elif count == 0:
                func_val = np.nan
            else:
                variables = [
                    col for col in ax_df.columns if ax_df[mask][col].nunique() > 1
                ]
                raise ValueError(
                    f"func_agg for {func_key} must be specified since multiple values "
                    f"per ({x_key}, {y_key})-pair are present. They vary in {variables}."
                )

            data[x_i, y_i] = func_val
            counts[x_i, y_i] = count

        # Use full_df to make sure vmin and vmax are consistent across all subplots
        func_min, func_max = self._get_func_limits(grid, full_df, func_key)
        norm = self.norm(vmin=func_min, vmax=func_max)

        pixel_alpha = counts / counts.max()

        ax.imshow(
            data.T,
            cmap=self.cmap,
            alpha=pixel_alpha.T,
            aspect="auto",
            norm=norm,
            interpolation="nearest",
            origin="lower",
        )

        if self.show_counts == "always":
            show_counts = True
        elif self.show_counts == ">1":
            show_counts = np.any(counts > 1)
        elif self.show_counts == "never":
            show_counts = False
        else:
            raise ValueError(f"Invalid show_counts value {self.show_counts!r}")

        if show_counts:
            for x_i, y_i in np.ndindex(shape):
                count = counts[x_i, y_i]

                if self.show_counts == ">1" and count <= 1:
                    continue

                pixel_color = self.cmap(norm(data[x_i, y_i]))
                pixel_color = apply_alpha(pixel_color, pixel_alpha[x_i, y_i])

                ax.text(
                    x_i,
                    y_i,
                    f"{counts[x_i, y_i]}",
                    ha="center",
                    va="center",
                    color=get_contrast_color(pixel_color),
                    fontsize=8,
                    transform=ax.transData,
                )

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

        for func_i, func_key in enumerate(grid.func_keys):
            # cax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
            func_min, func_max = self._get_func_limits(grid, full_df, func_key)

            mappable = plt.cm.ScalarMappable(
                norm=self.norm(vmin=func_min, vmax=func_max), cmap=self.cmap
            )
            label = grid.get_func_label(func_key)

            # Add colorbar to last column of each row corresponding to func_key
            start_row = func_i * n_rows_per_func
            end_row = (func_i + 1) * n_rows_per_func
            for ax in axs[start_row:end_row, -1]:
                fig.colorbar(mappable, ax=ax, label=label)

    def get_preferred_size(self, n_rows: int, n_cols: int):
        if n_rows == n_cols == 1:
            return 5, 4
        elif n_cols == 1:
            return 6, 3
        else:
            return 3, 3


Limits = tuple[float | None, float | None]


@as_converter
def plots_inferrer(plots: Iterable[Plot], owner: Any) -> tuple[Plot, ...]:
    plots = tuple(plots)

    if plots:
        return plots

    if owner is None:
        raise ValueError("Can only infer plot_type if owner is given")
    if not isinstance(owner, PlotGrid):
        raise ValueError(
            f"Can only infer plot_type of owner that is of type PlotGrid. Got {type(owner).__name__}"
        )

    if len(owner.arg_keys) == 1:
        return (LinePlot(),)
    elif len(owner.arg_keys) == 2:
        return (HeatmapPlot(),)
    else:
        raise ValueError(
            "Cannot infer plot_type from PlotGrid.arg_keys. Please specify it explicitly."
        )


# class Layer:
#     def draw()


@attrs.define(frozen=True)
class PlotGrid:
    arg_keys: tuple[str, ...] = attrs.field(converter=UniqueCollector(str))
    func_keys: tuple[str, ...] = attrs.field(converter=UniqueCollector(str))
    func_aggs: Mapping[str, str] = attrs.field(factory=dict, converter=MappingProxyType)
    plots: tuple[Plot, ...] = attrs.field(default=None, converter=plots_inferrer)
    col_keys: tuple[str, ...] = attrs.field(
        default=None, converter=UniqueCollector(str)
    )
    row_key: str | None = attrs.field(default=None)
    fig_key: str | None = attrs.field(default=None)
    fixed_values: Mapping[str, Any] = attrs.field(
        factory=dict, converter=MappingProxyType
    )
    mean_keys: tuple[str, ...] = attrs.field(
        default=None, converter=UniqueCollector(str)
    )
    agg_keys: tuple[str, ...] = attrs.field(
        default=None, converter=UniqueCollector(str)
    )
    indep_keys: tuple[str, ...] | None = attrs.field(
        default=None, converter=UniqueCollector(str).optional()
    )
    min_points_per_ax: int = attrs.field(default=2)
    key_format: Any = attrs.field(default=None)
    limits: dict[str, Limits] = attrs.field(factory=dict)
    whitelists: Mapping[str, tuple[Any, ...]] = attrs.field(
        factory=dict, converter=UniqueCollector().valuewise()
    )

    def _draw_func(
        self,
        df: pd.DataFrame,
        row_val: Any,
        col_vals: tuple,
        func_key: str,
        ax: Axes,
    ):
        plot_vals = {}
        if self.row_key:
            plot_vals[self.row_key] = row_val
        for col_key, col_val in zip(self.col_keys, col_vals):
            plot_vals[col_key] = col_val

        ax_df = df.copy()
        for key, value in plot_vals.items():
            ax_df = ax_df[ax_df[key] == value]

        func_agg = self.func_aggs.get(func_key, self.func_aggs.get("*"))

        for plot in self.plots:
            plot.draw_to_ax(self, df, ax, ax_df, func_key, func_agg)

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

    def _create_figure(
        self,
        df: pd.DataFrame,
        col_val_tuples: list[tuple],
        row_vals,
        func_keys: tuple,
    ):
        n_cols = len(col_val_tuples)
        n_rows = len(row_vals)
        n_funcs = len(func_keys)
        n_rows_total = n_rows * n_funcs

        width_per_plot, height_per_plot = self.plots[0].get_preferred_size(
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

        for row_i, col_i, func_i in np.ndindex(n_rows, n_cols, n_funcs):
            row_val = row_vals[row_i]
            col_vals = col_val_tuples[col_i]
            func_key = func_keys[func_i]
            ax = axs[row_i * n_funcs + func_i, col_i]
            self._draw_func(df, row_val, col_vals, func_key, ax)

        for plot in self.plots:
            plot.style_grid(self, fig, axs, df)

        return fig

    def _create_figure_df(
        self, df: pd.DataFrame, fig_val
    ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        indep_constants = (
            {key for key in self.indep_keys if df[key].nunique() <= 1}
            if self.indep_keys
            else set()
        )
        # print("INDEPENDENT CONSTANTS:", indep_constants)

        # Filter fig value
        if self.fig_key is not None:
            df = df[df[self.fig_key] == fig_val]

        # Filter fixed values
        for fixed_key, fixed_value in self.fixed_values.items():
            df = df[df[fixed_key] == fixed_value]

        # Apply whitelists
        for whitelist_key, whitelist in self.whitelists.items():
            df = df[df[whitelist_key].isin(whitelist)]

        # Apply limits
        for limit_key, (min_val, max_val) in self.limits.items():
            if min_val is not None:
                df = df[df[limit_key] >= min_val]
            if max_val is not None:
                df = df[df[limit_key] <= max_val]

        # Apply means
        for mean_key in self.mean_keys:
            df = df.groupby(mean_key).mean().reset_index()

        # Check if independent keys are constant
        df, fixed_values = self._resolve_unexpected_variables(
            df,
            constants=indep_constants,
        )

        return df, fixed_values

    def _resolve_unexpected_variables(
        self,
        df: pd.DataFrame,
        constants: set[str],
    ) -> tuple[pd.DataFrame, Mapping[str, Any]]:
        if self.indep_keys is None:
            return df, self.fixed_values

        fixed_values = dict(self.fixed_values)

        expected_variables = {
            *self.arg_keys,
            *self.func_keys,
            *self.col_keys,
            self.row_key,
            self.fig_key,
            *self.mean_keys,
            *self.agg_keys,
        }

        for plot in self.plots:
            expected_variables.update(plot.get_used_keys())

        for key in self.indep_keys:
            if key in expected_variables or key in constants:
                continue

            if key not in df.columns:
                print(f"Independent key {key!r} not in filtered DataFrame")
                continue

            # Independent key not used
            # -> check if it is constant and choose most frequent value if not
            if len(val_counts := df[key].value_counts()) > 1:
                most_freq_val = val_counts.idxmax()
                print(
                    f"Independent key {key!r} has multiple values: {val_counts.index.tolist()}. "
                    f"Using most frequent value: {most_freq_val}"
                )
                df = df[df[key] == most_freq_val]

            assert df[key].nunique() == 1

            fixed_values[key] = df[key].unique()[0]

        return df, fixed_values

    def show(self, df):
        n_samples_total = len(df)
        fig_vals = _get_val_list(df, self.fig_key) or [None]

        for fig_val in fig_vals:
            fig_df, fixed_values = self._create_figure_df(df, fig_val)

            col_val_tuples = list(
                itertools.product(
                    *(
                        _get_val_list(fig_df, col_key, self.min_points_per_ax) or [None]
                        for col_key in self.col_keys
                    )
                )
            )

            row_vals = _get_val_list(
                fig_df, self.row_key, self.min_points_per_ax, verbose=True
            ) or [None]

            fig = self._create_figure(fig_df, col_val_tuples, row_vals, self.func_keys)

            if self.fig_key is not None:
                fig.suptitle(f"{fmt(self.fig_key, self.key_format)} = {fmt(fig_val)}")

            fig.text(
                # center horizontally (using 0.5 in figure coordinates)
                0.5,
                # offset by 0.1 inch (divide by figure height to get inches)
                -0.1 / fig.get_figheight(),
                (
                    (
                        f"Mean over {fmt_intersection(self.mean_keys, self.key_format)}\n"
                        if self.mean_keys
                        else ""
                    )
                    + (
                        f"Fixed: {fmt_dict(fixed_values, key_format=self.key_format)}\n"
                        if fixed_values
                        else ""
                    )
                    + (
                        f"Aggregating over {fmt_intersection(self.agg_keys, self.key_format)}\n"
                        if self.agg_keys
                        else ""
                    )
                    + f"(contains {len(fig_df)} of {n_samples_total} samples)"
                    # token_pattern=r"(.+?\s*(?:,|and|\n)\s*)",
                ),
                ha="center",
                va="top",
                fontsize=8,
                style="italic",
                wrap=True,
            )

            plt.show()

    def get_func_label(self, func_key: str) -> str:
        label = fmt(func_key, self.key_format)

        if func_key in self.func_aggs:
            func_agg = self.func_aggs[func_key]
            label = f"{func_agg}({label})"

        return label
