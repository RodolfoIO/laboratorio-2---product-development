from __future__ import annotations

import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL no configurada")
def test_database_is_reachable() -> None:
    import asyncio

    from laboratorio_2.config import Settings
    from laboratorio_2.database import Database

    async def check() -> None:
        database = Database(Settings(database_url=os.environ["DATABASE_URL"]))
        try:
            await database.ping()
        finally:
            await database.close()

    asyncio.run(check())
