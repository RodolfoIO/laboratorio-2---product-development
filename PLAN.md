# Estado de implementación - Laboratorio 2

Actualizado: 2026-11-14

## Fase 0 - Fundaciones

- [x] Proyecto unificado en Python 3.12 con dependencias reproducibles.
- [x] Aplicación FastAPI con configuración por entorno y `lifespan`.
- [x] Paquete separado en contrato, limpieza, datos, persistencia, modelos y orquestación.
- [x] `.env.example`, `.gitignore`, configuración de pytest y Ruff.
- [x] Pruebas base y arranque de la aplicación.

## Fase 1 - Contrato y limpieza

- [x] Modelos Pydantic de entrada y respuesta.
- [x] Normalización de tiendas, familias, fechas, números y duplicados.
- [x] Políticas explícitas de reparar, advertir y rechazar.
- [x] Errores `422` con `detail` textual.
- [x] Respuestas con versión de modelo y observaciones de calidad.
- [x] Pruebas del contrato sin depender de MLflow.
- [x] Verificación con los ejemplos publicados `E01` a `E10`.

## Fase 2 - Persistencia

- [x] PostgreSQL del producto separado de Airflow y MLflow.
- [x] Esquema SQLAlchemy y migración Alembic inicial.
- [x] Registro de solicitudes, predicciones, correcciones y observaciones.
- [x] Observaciones idempotentes con conteo de cambios.
- [x] Ejecuciones administrativas persistidas.
- [x] Importación real de 10 tiendas, 12 familias, 83 052 ventas y 61 feriados.
- [x] Segunda importación verificada sin duplicar datos.
- [x] Prueba manual de observación idéntica con `registros_guardados: 0`.
- [x] Prueba automatizada de persistencia después de reiniciar el stack completo.

## Fase 3 - Datos, modelos y MLflow

- [x] Validación de los cinco CSV publicados.
- [x] Generación de lags y agregados sin usar ventas del mismo día.
- [x] Feriados nacionales, locales, comerciales y fechas de pago.
- [x] Baseline estacional y modelo global LightGBM.
- [x] Cuantil de inventario calculado desde los costos por familia.
- [x] Métricas MAE, RMSE, WAPE y costo de negocio.
- [x] Entrenamiento real: MAE de LightGBM 38.41 frente a 42.33 del baseline.
- [x] Costo de validación real: 92 172.40 frente a 357 496.70 del baseline.
- [x] Registro de baseline y LightGBM en MLflow con artefactos y linaje.
- [x] Alias `champion` y descarga del artefacto verificados.
- [x] Entrenamiento automático desde el servicio de bootstrap del Compose final.

## Fase 4 - Servicio de predicción

- [x] Carga local del modelo y respaldo cuando MLflow no responde.
- [x] Consulta del alias cada 30 segundos e intercambio atómico.
- [x] Pronóstico real con 12 familias y cantidad sugerida entera.
- [x] Persistencia del pronóstico y de sus correcciones.
- [x] `/modelo` y `/health` con estados `ok`, `degradado` y `503`.
- [x] Degradación correcta con MLflow y Airflow detenidos.
- [x] El endpoint de reentrenamiento devuelve `503` si Airflow no responde.
- [x] Cambio de modelo visible en menos de dos minutos mediante cambio de alias y polling.
- [x] Recuperación de salud en menos de dos minutos al reiniciar MLflow.

## Fase 5 - Airflow y reentrenamiento

- [x] DAG `novamercado_retrain` con máximo una ejecución activa y timeout de 10 minutos.
- [x] Validación previa y ejecución del entrenamiento desde el DAG.
- [x] Actualización de estado y protección del campeón ante fallos.
- [x] Endpoints de inicio y consulta implementados.
- [x] Autenticación compatible con Airflow 3.1 preparada.
- [x] Verificación de que el DAG aparece en Airflow y completa una ejecución real.
- [x] Verificación de promoción inicial, no reemplazo sin mejora y recarga automática.

## Fase 6 - Docker y operación

- [x] Imágenes fijas para API, Airflow, MLflow y PostgreSQL.
- [x] Construcción correcta de las tres imágenes del producto.
- [x] Tres PostgreSQL separadas y volúmenes persistentes definidos.
- [x] Servicio llamado exactamente `mlflow`.
- [x] Bootstrap y healthchecks implementados.
- [x] Arranque completo desde cero con `docker compose up`.
- [x] Reinicio sin repetir bootstrap ni perder datos.
- [x] Interrupción y recuperación de MLflow dentro de los límites exigidos.

## Fase 7 - Documentación y entrega

- [x] Suite local sin servidores externos.
- [x] Ruff configurado y sin errores en el código actual.
- [x] README completo con instalación y operación.
- [x] `DECISIONES.md` con alternativas y justificación.
- [x] `AGENTS.md` con contexto para futuros agentes.
- [x] Script de aceptación automatizado.
- [x] Evidencia final de MLflow, Airflow y respuestas de API.
- [ ] Tag `entrega-lab2` y último push de la versión validada.
