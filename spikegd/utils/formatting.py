from datetime import datetime
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping

import numpy as np

# NUMBERS


_DETECTABLE_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}


def _detect_constant(
    x: float, min_power=-1, max_power=1, max_int=100
) -> tuple[str, int, Fraction] | None:
    if np.isnan(x) or x == 0:
        return None

    from fractions import Fraction

    for name, value in _DETECTABLE_CONSTANTS.items():
        for power in range(min_power, max_power + 1):
            if power == 0:
                continue

            frac = Fraction(x / value**power)
            if frac.numerator < max_int and frac.denominator < max_int:
                return name, power, frac


def fmt_number(num: float, value_format: Any = ".3g"):
    if np.isnan(num):
        return "NaN"

    constant_result = _detect_constant(num)

    if constant_result is not None:
        constant, power, coeff = constant_result
        return f"{coeff ** power} {constant}^{power}"

    # Prevent integer numbers from being formatted with e-notation
    if num.is_integer() and num < 1e6:
        return str(int(num))

    return f"{num:{value_format}}"


# TIME


def fmt_timestamp(t: float):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H-%M-%S_%f")


def parse_timestamp(t: str):
    return datetime.strptime(t, "%Y-%m-%d_%H-%M-%S_%f").timestamp()


def fmt_duration(total_seconds: float):
    total_seconds = int(total_seconds)
    seconds = total_seconds % 60
    s = f"{seconds}s"

    total_minutes = total_seconds // 60
    if total_minutes == 0:
        return s

    minutes = total_minutes % 60
    s = f"{minutes}m {s}"

    total_hours = total_minutes // 60
    if total_hours == 0:
        return s

    hours = total_hours % 24
    s = f"{hours}h {s}"

    total_days = total_hours // 24
    if total_days == 0:
        return s

    days = total_days
    s = f"{days}d {s}"

    return s


# LISTS


def fmt_list(
    list_: Iterable,
    value_format=None,
    item_sep=", ",
    last_item_sep=" and ",
    empty="none",
):
    list_ = list(list_)

    if not list_:
        return empty

    if len(list_) == 1:
        return fmt(list_[0], value_format)

    return (
        item_sep.join(fmt(x, value_format) for x in list_[:-1])
        + last_item_sep
        + fmt(list_[-1], value_format)
    )


# DICTS


def fmt_dict(
    dict_: Mapping,
    value_format=None,
    key_format=None,
    item_sep=", ",
    value_sep=" = ",
    last_item_sep=" and ",
    empty="none",
):
    dict_ = dict(dict_)

    if not dict_:
        return empty

    items = [
        fmt(key, key_format) + value_sep + fmt(value, value_format)
        for key, value in dict_.items()
    ]

    if len(items) == 1:
        return items[0]

    return item_sep.join(items[:-1]) + last_item_sep + items[-1]


def fmt_dict_multiline(d: dict, value_format=None, key_format=None, indent=26):
    lines = []

    for k, v in d.items():
        v_str = fmt(v, value_format)

        # indent
        v_str = v_str.replace("\n", "\n" + " " * indent)

        k_str = fmt(k, key_format)

        lines.append(f"{k_str:<{indent - 1}} {v_str}")

    return "\n".join(lines)


def print_dict(d: dict, value_format=None, key_format=None, indent=26):
    print(fmt_dict_multiline(d, value_format, key_format, indent))


# MISC


def fmt_plural(count: int, singular: str, plural: str = "{}s", zero: str = "{}s"):
    if count == 0:
        return zero.format(singular)
    elif count == 1:
        return singular
    else:
        return plural.format(singular)


def fmt_count(count: int, singular: str, plural: str = "{}s", zero: str = "{}s"):
    return f"{count} {fmt_plural(count, singular, plural, zero)}"


# GENERAL

Formatter = Callable[..., str]

_formatters: dict[type, Formatter] = {
    int: fmt_number,
    float: fmt_number,
    np.number: fmt_number,
    dict: fmt_dict,
    list: fmt_list,
    tuple: fmt_list,
    set: fmt_list,
}


def fmt(
    value,
    formatter: Formatter | str | type | None = None,
    **options,
) -> str:
    if formatter is None:
        for type_, formatter in _formatters.items():
            if isinstance(value, type_):
                return fmt(value, formatter, **options)

        return str(value)
    elif isinstance(formatter, str):
        return formatter.format(value)
    elif isinstance(formatter, type):
        return fmt(value, _formatters.get(formatter), **options)
    elif callable(formatter):
        return formatter(value, **options)
    else:
        raise ValueError(f"Invalid formatter {formatter}")
