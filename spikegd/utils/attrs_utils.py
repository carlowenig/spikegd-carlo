import inspect
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, cast

import attrs

from .formatting import fmt_union


class Converter[A, B](ABC, attrs.Converter):
    def __init__(self):
        super().__init__(self, takes_self=True)

    @abstractmethod
    def _convert(self, obj: A, owner: Any) -> B: ...

    def __call__(self, obj: A, owner: Any = None, field: Any = None, /) -> B:
        return self._convert(obj, owner)

    def __or__[C](self, other: "ConverterLike[B, C]") -> "Converter[A, C]":
        return Compose(self, other)

    def __ror__[C](self, other: "ConverterLike[C, A]") -> "Converter[C, B]":
        return Compose(other, self)

    def optional(self):
        return OptionalConverter(self)

    def valuewise(self):
        return Valuewise(self)

    def elementwise(self):
        return Elementwise(self)


class CustomConverter[A, B](Converter[A, B]):
    def __init__(self, func: Callable[[A, Any], B] | Callable[[A], B]) -> None:
        self.func = func
        self._accepts_owner = len(inspect.signature(func).parameters) >= 2
        super().__init__()

    def _convert(self, obj: A, owner: Any) -> B:
        if self._accepts_owner:
            return self.func(obj, owner)  # type: ignore
        else:
            return self.func(obj)  # type: ignore


type ConverterLike[A, B] = Callable[[A], B] | Callable[[A, Any], B] | Converter[A, B]


def as_converter[A, B](obj: ConverterLike[A, B]) -> Converter[A, B]:
    if isinstance(obj, Converter):
        return obj
    else:
        return CustomConverter(obj)


class Compose[A, B, C](Converter[A, C]):
    def __init__(self, first: ConverterLike[A, B], second: ConverterLike[B, C]) -> None:
        self.first = as_converter(first)
        self.second = as_converter(second)
        super().__init__()

    def _convert(self, obj: A, owner: Any) -> C:
        return self.second(self.first(obj, owner), owner)


_default_collection_types: tuple[type[Iterable], ...] = (tuple, list, set)

type Many[T] = tuple[T, ...] | list[T] | set[T] | T | None


class Collector[T](Converter[Many[T], tuple[T, ...]]):
    def __init__(
        self,
        element_types: type[T] | tuple[type[T], ...] | None = None,
        collection_types: tuple[type[Iterable], ...] = _default_collection_types,
    ) -> None:
        self.element_types: tuple[type[T], ...] = (
            ()
            if element_types is None
            else element_types
            if isinstance(element_types, tuple)
            else (element_types,)
        )
        self.collection_types = collection_types
        super().__init__()

    def _convert(
        self,
        obj: Many[T],
        owner: Any,
    ) -> tuple[T, ...]:
        if obj is None:
            return ()

        if self.element_types:
            if isinstance(obj, self.element_types):
                return (obj,)
            elif isinstance(obj, self.collection_types):
                unchecked = tuple(obj)
                # check if all elements are of the correct type
                incorrect = [
                    el for el in unchecked if not isinstance(el, self.element_types)
                ]
                if incorrect:
                    raise TypeError(
                        f"Expected all elements to be of type "
                        f"{fmt_union(self.element_types)}, found {incorrect}"
                    )
                return unchecked
            else:
                expected_types = self.element_types + self.collection_types + (None,)
                raise TypeError(
                    f"Expected {fmt_union(expected_types)}, got {type(obj).__name__}"
                )
        else:
            # no element types given -> no type checking
            if isinstance(obj, (list, tuple, set)):
                return tuple(obj)
            else:
                return (obj,)


def collect[T](
    elements: Many[T],
    element_type: type[T] | None = None,
    collection_types: tuple[type[Iterable], ...] = _default_collection_types,
) -> tuple[T, ...]:
    return Collector[T](element_type, collection_types)(elements)


class Deduplicater[T](Converter[Iterable[T], tuple[T, ...]]):
    def __init__(self, __generic_type_helper: type[T] | None = None) -> None:
        super().__init__()

    def _convert(self, obj: Iterable[T], owner: Any) -> tuple[T, ...]:
        return tuple(dict.fromkeys(obj))


def deduplicate[T](elements: Iterable[T]) -> tuple[T, ...]:
    return Deduplicater[T]()(elements)


class UniqueCollector[T](Compose[Many[T], tuple[T, ...], tuple[T, ...]]):
    def __init__(
        self,
        element_type: type[T] | None = None,
        collection_types: tuple[type[Iterable], ...] = _default_collection_types,
    ) -> None:
        self.collector = Collector(element_type, collection_types)
        self.deduplicater = Deduplicater(element_type)
        super().__init__(self.collector, self.deduplicater)


def collect_unique[T](
    elements: Many[T],
    element_type: type[T] | None = None,
    collection_types: tuple[type[Iterable], ...] = _default_collection_types,
) -> tuple[T, ...]:
    return UniqueCollector[T](element_type, collection_types)(elements)


class Elementwise[A, B](Converter[Iterable[A], tuple[B, ...]]):
    def __init__(self, element_converter: ConverterLike[A, B]) -> None:
        self.element_converter = as_converter(element_converter)
        super().__init__()

    def _convert(self, obj: Iterable[A], owner: Any) -> tuple[B, ...]:
        return tuple(self.element_converter(el) for el in obj)


class Valuewise[A, B](Converter[Mapping[str, A], Mapping[str, B]]):
    def __init__(self, value_converter: ConverterLike[A, B]) -> None:
        self.value_converter = as_converter(value_converter)
        super().__init__()

    def _convert(self, obj: Mapping[str, A], owner: Any) -> Mapping[str, B]:
        return MappingProxyType({k: self.value_converter(v) for k, v in obj.items()})


class Caster[A, B](Converter[A, B]):
    def __init__(self, type_: type[B] | None = None, check=True) -> None:
        self.type = type_
        self.check = check
        super().__init__()

    def _convert(self, obj: A, owner: Any) -> B:
        if self.check and self.type is not None and not isinstance(obj, self.type):
            raise TypeError(f"Expected {self.type.__name__}, got {type(obj).__name__}")

        return cast(B, obj)


class OptionalConverter[A, B](Converter[A | None, B | None]):
    def __init__(self, inner: ConverterLike[A, B]) -> None:
        self.inner = as_converter(inner)
        super().__init__()

    def _convert(self, obj: A | None, owner: Any) -> B | None:
        if obj is None:
            return None
        else:
            return self.inner(obj, owner)
