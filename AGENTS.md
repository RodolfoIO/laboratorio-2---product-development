# Guia para agentes de programacion

## Estructura

- `src/laboratorio_2/`: paquete unico del producto.
- `tests/`: contrato, limpieza, features, modelos y pruebas de integracion.
- `dags/`: DAG de Airflow; debe orquestar mediante HTTP y no importar SQLAlchemy 2.x.
- `migrations/`: esquema Alembic de la base del producto.
- `data/raw/`: datos publicados, fuera de Git e imagenes.
- `docs/evidence/`: evidencia reproducible de operacion.

## Comandos

```bash
uv sync
uv run pytest
uv run ruff check src tests dags migrations
docker compose config --quiet
docker compose up --build
```

## Invariantes

- No usar el reloj del sistema para validar fechas de dominio.
- No permitir que una prediccion use ventas de la misma fecha o posteriores.
- Mantener los nombres y tipos del contrato del PDF.
- No promover un candidato que no supere el criterio documentado.
- No cambiar el alias `champion` cuando el entrenamiento falla.
- Mantener la base del producto separada de Airflow y MLflow.
- No versionar CSV publicados, modelos, secretos ni volumenes.

## Pruebas obligatorias antes de integrar

- Contrato valido y defectuoso sin MLflow en ejecucion.
- Idempotencia de observaciones.
- Ausencia de fuga temporal.
- Serializacion de artefactos y politica de promocion.
- Carga del DAG sin errores de importacion.
- `GET /health` en estados `ok`, `degradado` y sin modelo.
