# Decisiones tecnicas y de producto

## Defensa del contrato

| Problema | Politica | Razon |
|---|---|---|
| `tienda_id` como texto, `"005"` o `"T05"` | Reparar al ID del catalogo | La transformacion es determinista y el ID existe |
| Familia con mayusculas, acentos, espacios o nombre visible | Reparar al `familia_id` canonico | Se resuelve contra ID o nombre del catalogo |
| Entero enviado como texto o decimal integral | Reparar | No cambia el valor numerico |
| Numero con separador de miles o coma decimal | Reparar | `1,292` y `50,0` tienen interpretacion inequivoca |
| Fecha alterna o timestamp | Reparar si es inequivoca | El dia resultante es unico |
| Fecha `DD/MM` y `MM/DD` igualmente posible | Rechazar con `422` | Elegir una interpretacion inventaria informacion |
| `en_promocion` ausente | Advertir y usar `0` | Permite responder e informar explicitamente el supuesto |
| Familias del catalogo ausentes | Advertir; responder solo por las presentes | No se agregan predicciones que la tienda no solicito |
| Duplicado identico | Reparar eliminando la repeticion | Misma familia y mismo valor |
| Duplicado con valores contradictorios | Rechazar con `422` | No existe una eleccion segura |
| ID desconocido, campo renombrado o ventas negativas | Rechazar con `422` | No hay una reparacion sin inventar datos |

No se usa coincidencia difusa para familias o tiendas. Una heuristica de distancia podria asignar una solicitud a un catalogo incorrecto.

## Informacion usada por el modelo

El dataset de entrenamiento se ordena por tienda, familia y fecha. Las variables son calendario, dia de la semana, feriados nacionales y locales, fechas comerciales, pagos, promociones, lags de 1/7/14/28 dias y medias y medianas moviles de 7/28 dias.

La construccion usa `shift` antes de los agregados moviles. Al servir una fecha, se descartan las ventas de esa fecha y posteriores. Para observaciones nuevas, `en_promocion` se recupera de la solicitud de pronostico mas reciente de la misma tienda, familia y fecha.

La validacion es temporal: los ultimos 28 dias del conjunto forman el holdout. No se usa una division aleatoria porque romperia la secuencia temporal.

## Baseline y modelo

El baseline calcula medianas estacionales por tienda, familia y dia de la semana, con fallback a nivel familia y global. LightGBM entrena un modelo global de regresion y modelos de cuantiles para cada razon de costo presente en las familias.

En la validacion con los datos publicados:

| Modelo | MAE | RMSE | WAPE | Costo de negocio |
|---|---:|---:|---:|---:|
| Baseline | 42.33 | 80.19 | 0.1980 | 357 496.70 |
| LightGBM | 38.41 | 70.47 | 0.1797 | 92 172.40 |

## Cantidad sugerida

Para cada familia:

```text
q = costo_faltante / (costo_faltante + costo_sobrante)
```

La cantidad se obtiene del cuantil `q` de la distribucion condicional de ventas y se redondea hacia arriba a un entero no negativo. Esta es la solucion discreta del problema de inventario con costo de faltante y sobrante asimetricos.

## Criterio de reemplazo

Un candidato se promueve solo si:

- Reduce el costo de validacion al menos 1 % frente al campeon.
- Su MAE no empeora mas de 5 % frente al campeon.

Si no existe campeon, se promueve el mejor entre baseline y LightGBM. Si falla la carga del campeon, el reentrenamiento falla y no modifica el alias. Un empate conserva el modelo vigente.

Las dos condiciones evitan sacrificar calidad tecnica por una mejora economica marginal. Los umbrales son parametros de configuracion y la respuesta incluye el motivo con los costos comparados.

## MLOps

MLflow registra un run por baseline y otro por candidato. Cada artefacto incluye el modelo serializado, metricas, cutoff, huella del dataset y revision del codigo. El registro `novamercado-forecast` usa el alias `champion`.

Airflow 3.1 depende internamente de SQLAlchemy 1.4, mientras el servicio usa SQLAlchemy 2.x. Para evitar mezclar entornos incompatibles, el DAG actua como orquestador HTTP: valida disponibilidad de datos y solicita el trabajo a la API. La API ejecuta el entrenamiento en un hilo con un pool de base de datos propio y sigue atendiendo pronosticos.

## Resiliencia

La API conserva el modelo en memoria y en `model-cache`. La falta de MLflow no impide predecir; la falta de base de datos si impide confirmar una operacion porque no se puede garantizar el registro. `GET /health` verifica base de datos, MLflow, Airflow y modelo.

Un modelo promovido se detecta con polling cada 30 segundos. El archivo se descarga a un temporal, se valida y se reemplaza atomicamente antes de cambiar la referencia en memoria.

## Persistencia

Cada PostgreSQL usa un volumen nombrado independiente. Las observaciones tienen una clave unica por tienda, familia y fecha. El upsert solo cuenta una insercion o un cambio real; un reenvio identico devuelve cero.

## Alternativas descartadas

- SQLite para la operacion: no modela adecuadamente la concurrencia y persistencia esperadas.
- Base unica para producto, Airflow y MLflow: el documento exige separar la base de Airflow y complica aislar fallos.
- Entrenamiento dentro de la event loop del API server: bloquea pronosticos durante el ajuste de modelos.
- Promover siempre la ultima version: no cumple el criterio automatico ni protege al campeon.
- Agregar familias ausentes a la respuesta: inventaria el alcance de una solicitud defectuosa.
