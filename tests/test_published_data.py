from __future__ import annotations

from pathlib import Path

import pytest

from laboratorio_2.data_import import REQUIRED_FILES, load_published_data

DATA_DIR = Path("data/raw")


@pytest.mark.skipif(
    not all((DATA_DIR / filename).is_file() for filename in REQUIRED_FILES.values()),
    reason="Los archivos publicados no estan en data/raw",
)
def test_published_data_has_expected_shape() -> None:
    bundle = load_published_data(DATA_DIR)
    assert len(bundle.stores) == 10
    assert len(bundle.families) == 12
    assert len(bundle.history) == 83_052
    assert len(bundle.holidays) == 61
    assert bundle.catalog.min_date.isoformat() == "2024-03-01"
    assert bundle.catalog.max_date.isoformat() == "2026-12-31"
