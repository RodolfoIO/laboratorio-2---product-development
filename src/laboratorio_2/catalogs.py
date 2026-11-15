from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any


def normalize_key(value: Any) -> str:
    if value is None:
        raise ValueError("no puede estar vacio")
    normalized = unicodedata.normalize("NFKD", str(value).strip())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


@dataclass(frozen=True)
class Store:
    id: int
    name: str
    city: str
    store_format: str


@dataclass(frozen=True)
class Family:
    id: str
    name: str


@dataclass(frozen=True)
class Cost:
    family_id: str
    shortage: float
    surplus: float

    @property
    def critical_ratio(self) -> float:
        total = self.shortage + self.surplus
        if total <= 0:
            return 0.5
        return min(max(self.shortage / total, 0.0), 1.0)


@dataclass
class Catalog:
    stores: dict[int, Store] = field(default_factory=dict)
    families: dict[str, Family] = field(default_factory=dict)
    costs: dict[str, Cost] = field(default_factory=dict)
    min_date: date | None = None
    max_date: date | None = None
    _family_lookup: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.rebuild_lookup()

    def rebuild_lookup(self) -> None:
        lookup: dict[str, str] = {}
        for family in self.families.values():
            for candidate in (family.id, family.name):
                key = normalize_key(candidate)
                if not key:
                    continue
                existing = lookup.get(key)
                if existing is not None and existing != family.id:
                    raise ValueError(f"alias de familia ambiguo: {candidate!r}")
                lookup[key] = family.id
        self._family_lookup = lookup

    def resolve_family(self, value: Any) -> tuple[str, bool]:
        key = normalize_key(value)
        family_id = self._family_lookup.get(key)
        if family_id is None:
            raise KeyError(f"familia desconocida: {value!r}")
        return family_id, str(value).strip() != family_id

    def resolve_store(self, value: Any) -> tuple[int, bool]:
        if isinstance(value, bool):
            raise ValueError("tienda_id no puede ser booleano")
        if isinstance(value, int):
            store_id = value
            repaired = False
        elif isinstance(value, float) and value.is_integer():
            store_id = int(value)
            repaired = True
        elif isinstance(value, str):
            candidate = value.strip()
            match = re.fullmatch(r"(?:[Tt]\s*)?0*(\d+)", candidate)
            if match is None:
                raise ValueError(f"tienda_id invalido: {value!r}")
            store_id = int(match.group(1))
            repaired = True
        else:
            raise ValueError(f"tienda_id invalido: {value!r}")
        if store_id not in self.stores:
            raise KeyError(f"tienda desconocida: {store_id}")
        return store_id, repaired

    def validate_date_range(self, value: date) -> None:
        if self.min_date is not None and value < self.min_date:
            raise ValueError(f"fecha fuera del historico soportado: {value.isoformat()}")
        if self.max_date is not None and value > self.max_date:
            raise ValueError(f"fecha fuera del calendario soportado: {value.isoformat()}")

    @property
    def family_ids(self) -> list[str]:
        return sorted(self.families)
