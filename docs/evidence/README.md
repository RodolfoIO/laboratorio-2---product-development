# Evidencia de funcionamiento

Fecha de verificacion: 2026-11-14, zona `America/Guatemala`.

## Datos importados

```text
Datos importados: 10 tiendas, 12 familias
Historico: 83 052 ventas
Feriados y fechas especiales: 61
Huella del paquete publicado: c31148a720b8
```

Una segunda ejecucion de `import-data` conservo los mismos conteos, sin duplicar historia.

## Modelos

Metricas de validacion temporal:

| Version | Algoritmo | MAE | Costo |
|---:|---|---:|---:|
| 1 | Baseline | 42.33 | 357 496.70 |
| 2 | LightGBM | 38.41 | 92 172.40 |

El alias `champion` apunta a la version `2`. Una descarga desde MLflow produjo un artefacto LightGBM cargable por el servicio.

## DAG

El DAG `novamercado_retrain` cargo sin errores de importacion. Ejecucion final:

```json
{
  "run_id": "06b34103-bdff-41ce-8dfe-20e4e3c29bae",
  "state": "success",
  "start_date": "2026-11-15T03:49:52.193227+00:00",
  "end_date": "2026-11-15T03:50:27.179567+00:00"
}
```

Duracion aproximada: 35 segundos.

La aceptacion automatizada posterior (`scripts/acceptance.py --reentrenar`) completo
salud, contrato, rechazo `422`, idempotencia y otro reentrenamiento exitoso en
aproximadamente 40 segundos, conservando la version 2 del modelo.

Respuesta final del endpoint:

```json
{
  "estado": "exitoso",
  "promovido": false,
  "version": "2",
  "motivo": "Costo en validacion 92172.40 vs 92172.40 del vigente; MAE 38.41 vs 38.41. No alcanzo la mejora minima de costo de 1.0%."
}
```

## Respuesta valida

El ejemplo publicado de la tienda 5 produjo 12 predicciones con `calidad.estado = ok`, `version_modelo = 2` y cantidades sugeridas enteras.

La observacion publicada respondio inicialmente `registros_guardados = 12`; el reenvio identico respondio `0`.

## Errores controlados

- Duplicado contradictorio: `422`, con detalle textual.
- Ventas negativas: `422`.
- Numero `"1,292"`: reparado a `1292`.
- Numero `"50,0"`: reparado a `50`.
- Tienda `"T05"`: reparada a `5`.
- Familia `"LÁCTEOS"`: reparada a `lacteos`.

## Interrupcion de MLflow

Con MLflow detenido:

- `GET /health` respondio `200` con `estado = degradado`.
- `POST /v1/prediccion` siguio respondiendo `200` con la version 2.
- Al reiniciar MLflow, `GET /health` volvio a `estado = ok`.

## Reinicio completo

Se detuvo todo el stack sin borrar volumenes y se ejecuto de nuevo `docker compose up -d`.

- El bootstrap termino sin reentrenar.
- El modelo siguio siendo la version `2`.
- La observacion marcadora `(tienda 5, abarrotes, 2025-11-18, ventas 777)` sobrevivio.
- `GET /health` volvio a `ok`.
