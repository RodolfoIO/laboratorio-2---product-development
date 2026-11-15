# NovaMercado - Servicio de pronostico diario

Servicio de machine learning en produccion para pronosticar ventas diarias por tienda y familia, registrar ventas reales y reentrenar el modelo con Airflow y MLflow.

## Arquitectura

- `api`: FastAPI en el puerto `8000`, fallback local del modelo y validacion del contrato.
- `postgres-producto`: solicitudes, predicciones, correcciones, observaciones y ejecuciones.
- `mlflow`: experimentos, artefactos y Model Registry en el puerto `5000`.
- `airflow-*`: orquestacion del DAG `novamercado_retrain` en el puerto `8080`.
- `postgres-airflow` y `postgres-mlflow`: bases separadas de la base del producto.
- `bootstrap`: importa los datos publicados y crea el primer modelo solo cuando no existe.

## Requisitos

- Docker con Docker Compose.
- `uv` para desarrollo y pruebas locales.
- Los archivos publicados por el curso.

## Archivos publicados

Copie los archivos en `data/raw/`, conservando estos nombres:

```text
data/raw/
├── tiendas.csv
├── familias.csv
├── historico_ventas.csv
├── feriados.csv
├── costos.csv
├── catalogo_errores.md
└── ejemplos/
```

Los CSV no se versionan en Git ni se copian dentro de las imagenes. `data/raw/ejemplos/` se usa para pruebas de aceptacion.

## Levantar desde cero

```bash
cp .env.example .env
docker compose up --build
```

No se necesitan comandos adicionales para importar datos, aplicar migraciones, entrenar el primer modelo o cargar el alias `champion`.

Servicios:

| Servicio | URL |
|---|---|
| API y OpenAPI | http://localhost:8000 y http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

Credenciales locales iniciales de Airflow: `admin` / `admin`. Cambielas en `.env` antes de exponer el producto fuera del equipo local.

La primera construccion no cuenta para el limite de 10 minutos. En arranques posteriores, el bootstrap detecta el modelo y los datos existentes y no repite el trabajo.

## API

```bash
curl -X POST http://localhost:8000/v1/prediccion \
  -H 'Content-Type: application/json' \
  --data-binary @data/raw/ejemplos/prediccion_valida.json

curl -X POST http://localhost:8000/v1/observaciones \
  -H 'Content-Type: application/json' \
  --data-binary @data/raw/ejemplos/observacion_valida.json

curl http://localhost:8000/modelo
curl http://localhost:8000/health
```

Reentrenamiento:

```bash
curl -X POST http://localhost:8000/admin/reentrenar
curl http://localhost:8000/admin/reentrenar/REEMPLACE-POR-EL-ID
```

`POST /admin/reentrenar` responde `202` con un identificador o `503` si Airflow no puede iniciar el DAG. La API consulta el alias `champion` cada 30 segundos, por lo que un modelo promovido se carga sin reiniciar el proceso.

## Desarrollo y pruebas

```bash
uv sync
uv run pytest
uv run ruff check src tests dags migrations
```

La suite principal no requiere MLflow ni Airflow en ejecucion. Las pruebas marcadas como `integration` usan `DATABASE_URL` cuando esta definida.

Para ejecutar la aceptacion contra el stack:

```bash
uv run python scripts/acceptance.py \
  --base-url http://localhost:8000 \
  --data-dir data/raw \
  --reentrenar
```

## Datos y tiempo

- El servicio no usa el reloj del sistema para decidir si una fecha es valida.
- Las fechas se validan contra el historico y `feriados.csv`.
- Al pronosticar una fecha, solo se usan ventas con fecha anterior.
- Las promociones se toman de la solicitud de pronostico.
- La cantidad sugerida usa el cuantil que minimiza los costos por familia.

## Persistencia

Volumenes principales:

- `postgres-product-data`
- `postgres-airflow-data`
- `postgres-mlflow-data`
- `mlflow-artifacts`
- `model-cache`
- `airflow-logs`

`docker compose down` conserva los volumenes. `docker compose down -v` elimina toda la persistencia y debe usarse solo para reiniciar intencionalmente el laboratorio desde cero.

## Diagnostico

```bash
docker compose ps
docker compose logs -f api bootstrap airflow-scheduler
docker compose exec airflow-dag-processor airflow dags list-import-errors
docker compose exec airflow-scheduler \
  airflow dags list-runs novamercado_retrain -o table
```

Si `GET /health` devuelve `degradado`, el campo `motivo` identifica la dependencia. Si no existe un modelo, responde `503`.

## Entrega

El tag `entrega-lab2` apunta a la version validada. Consulte `PLAN.md` para el estado de implementacion y `DECISIONES.md` para las justificaciones tecnicas.
