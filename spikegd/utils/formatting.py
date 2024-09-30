import re
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

            coeff = Fraction(x / value**power)
            if coeff.numerator < max_int and coeff.denominator < max_int:
                return name, power, coeff


def fmt_constant_with_coeff(name, power, coeff):
    if coeff == 0:
        return "0"

    s = ""

    if coeff != 1 or power < 0:
        s += f"{coeff} "

    if power == 1:
        s += name
    elif power == -1:
        s += f"/ {name}"
    elif power < 0:
        s += f"/ {name}^{-power}"
    else:
        s += f"{name}^{power}"

    return s


def fmt_number(num: float, value_format: Any = ".3g"):
    if np.isnan(num):
        return "NaN"

    constant_result = _detect_constant(num)

    if constant_result is not None:
        return fmt_constant_with_coeff(*constant_result)

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
    items: Iterable,
    value_format=None,
    item_sep=", ",
    last_item_sep=", ",
    empty="none",
):
    items = list(items)

    if not items:
        return empty

    if len(items) == 1:
        return fmt(items[0], value_format)

    return (
        item_sep.join(fmt(x, value_format) for x in items[:-1])
        + last_item_sep
        + fmt(items[-1], value_format)
    )


def fmt_intersection(
    items: Iterable,
    value_format=None,
):
    return fmt_list(
        items,
        value_format,
        last_item_sep=" and ",
        empty="none",
    )


def fmt_union(
    items: Iterable,
    value_format=None,
):
    return fmt_list(items, value_format, last_item_sep=" or ", empty="any")


# DICTS


def fmt_dict(
    dict_: Mapping,
    value_format=None,
    key_format=None,
    item_sep=", ",
    value_sep=" = ",
    last_item_sep=", ",
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


def fmt_type(t: type):
    return t.__name__


Formatter = Callable[..., str]

_formatters: dict[type, Formatter] = {
    int: fmt_number,
    float: fmt_number,
    np.number: fmt_number,
    dict: fmt_dict,
    list: fmt_list,
    tuple: fmt_list,
    set: fmt_list,
    type: fmt_type,
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


def wrap_text(
    s: str | Iterable[str],
    width: int = 80,
    allow_token_split=True,
    trim_line_end=True,
    token_pattern=r"(.+?(?:\s+|[,;\-]\s*|[.?!:]\s+))",
) -> str:
    if isinstance(s, str):
        # split by whitespace, punctuation, etc.
        tokens = re.split(token_pattern, s)
    else:
        tokens = list(s)

    lines = [""]

    for token in tokens:
        if len(lines[-1]) + len(token) <= width:
            # token fits on the current line
            lines[-1] += token
            continue

        # token does not fit -> add new line

        if allow_token_split:
            # force wrapping -> split token if too long
            while True:
                lines.append(token[:width])
                token = token[width:]

                if not token:
                    break
        else:
            # no force wrapping -> append token despite overflow
            lines.append(token)

    if trim_line_end:
        lines = [line.rstrip() for line in lines]

    return "\n".join(lines)
