from __future__ import annotations

import inspect
from typing import Any, Callable, ClassVar, Self, overload


class Field[T]:
    def _get_value(self, model: Model) -> T:
        return model._values[self.name]

    def _set_value(self, model: Model, value: T) -> None:
        model._values[self.name] = value

    def __init__(self, name: str | None = None):
        self._name = name
        self._owner: type[Model] | None = None

    def __set_name__(self, owner: type[Model], name: str):
        self._name = name
        self._owner = owner

    @property
    def name(self):
        if self._name is None:
            raise ValueError("Field name not set")
        return self._name

    @property
    def owner(self):
        if self._owner is None:
            raise ValueError("Field owner not set")
        return self._owner

    @overload
    def __get__(self, instance: None, owner: type[Model]) -> Self: ...

    @overload
    def __get__(self, instance: Model, owner: type[Model]) -> T: ...

    def __get__(self, instance: Model | None, owner: type[Model]) -> Self | T:
        if instance is None:
            return self
        return self._get_value(instance)

    def __set__(self, instance: Model, value: T) -> None:
        self._set_value(instance, value)


class Model:
    _fields: ClassVar[dict[str, Field[Any]]]

    def __init__(self, **values):
        self._values = {}

        for key, value in values.items():
            setattr(self, key, value)

        for field in self._fields.values():
            if field.name not in self._values:
                self._values[field.name] = field._get_value(self)

    def __init_subclass__(cls) -> None:
        cls._fields = {}

        for name, attr in cls.__dict__.items():
            if not isinstance(attr, Field):
                continue

            if attr._owner is not cls:
                raise ValueError(f"Field {attr.name} has wrong owner")

            if attr.name != name:
                raise ValueError(f"Field {attr.name} has wrong name")

            cls._fields[name] = attr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{name}={getattr(self, name)!r}' for name in self._fields)})"


class Computed[T](Field[T]):
    def __init__(self, compute: Callable[..., T]):
        super().__init__()
        self._compute = compute

        self._dependencies = [
            param.name
            for param in inspect.signature(compute).parameters.values()
            if param.name != "self"
        ]

        for dependency in self._dependencies:
            if dependency not in self.owner._fields:
                raise ValueError(f"Field {dependency} not found")

    def _get_value(self, model: Model) -> T:
        return self._compute(model)

    def _set_value(self, model: Model, value: T) -> None:
        raise ValueError("Computed field cannot be set")


class Test(Model):
    a = Field[int]()
    b = Computed[int](lambda a: a + 1)


test = Test(a=1)

print(test)
