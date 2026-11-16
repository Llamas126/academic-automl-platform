# academic-automl-platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/Python-%3E%3D3.13-3776AB.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

Backend asíncrono de AutoML para investigación académica: ingesta de tablas CSV/XLSX, modelado multi-target (regresión y clasificación mixtas), comparación de algoritmos con validación cruzada anidada, optimización de hiperparámetros con Optuna, explicabilidad SHAP y entrega de un paquete ZIP listo para publicación (reporte PDF/LaTeX, tablas booktabs, BibTeX, figuras 300 DPI PNG+EPS, modelos `.joblib` y predicciones out-of-fold).

---

> **English:** Async AutoML backend for scientific research and academic publication: CSV/XLSX table ingestion, multi-target modeling (mixed regression and classification), algorithm comparison with nested cross-validation, hyperparameter optimization with Optuna, SHAP explainability, and delivery of a publication-ready ZIP bundle (PDF/LaTeX report, booktabs tables, BibTeX, 300 DPI PNG+EPS figures, `.joblib` models, and out-of-fold predictions).

## Arquitectura

Microservicios desacoplados que comparten un paquete común (`packages/shared/automl_shared`) y un volumen de datos; el punto de entrada único es el gateway:

| Servicio | Paquete | Responsabilidad |
|---|---|---|
| **gateway** | `services/gateway` | Proxy HTTP a los servicios internos, API key, rate limit (Redis sliding-window), security headers |
| **ingestion** | `services/ingestion` | Subida CSV/XLSX, profiling y sugerencia de targets |
| **mlcore** | `services/ml-core` | Entrenamiento (Optuna, CV anidada, multi-target), orquestación de la cadena Celery `train → explain → export`, limpieza TTL (beat) |
| **explainability** | `services/explainability` | Figuras SHAP desde el contrato de entrenamiento persistido |
| **exporter** | `services/exporter` | Bundle científico: figuras, tablas LaTeX, BibTeX, reporte PDF/Tectonic, ZIP+MANIFEST |
| **inference** | `services/inference` | Predicción con los modelos `.joblib` de un run |

- **Contrato entre servicios**: mlcore persiste en `data/jobs/<id>/work` (`training_meta.json`, `background.csv`, `eval_*.npz`) y `out` (métricas, modelos firmados); explainability y exporter son consumidores puros de esos archivos , no se conocen entre sí
- **Colas Celery**: `ml.training`, `explain.shap`, `export.package`, `maintenance`; guardia de etapas evita propagación cuando un eslabón falla
- **Compatibilidad**: el monolito composable sigue disponible para desarrollo/tests vía los shims de `app/` (re-exportan los paquetes de servicio); la suite tier1 corre contra él
- **Seguridad**: opt-in en el gateway (`services/gateway/gateway/middleware/security.py`): API key para mutaciones y rate limit Redis con fallback en memoria

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/v1/health` | Liveness |
| POST | `/api/v1/datasets/upload` | Sube CSV/XLSX, devuelve profiling y sugerencias de targets (201) |
| GET | `/api/v1/datasets/{dataset_id}` | Metadatos del dataset |
| DELETE | `/api/v1/datasets/{dataset_id}` | Borra dataset |
| POST | `/api/v1/train` | Crea job de entrenamiento (202) |
| POST | `/api/v1/predict/{job_id}` | Inferencia con los modelos de un run terminado: JSON `{"records":[...]}` o upload CSV/XLSX; features temporales reconstruidas desde crudas + historial; query opcional `?targets=t1,t2&horizon=3&mode=recursive` |
| GET | `/api/v1/status/{job_id}` | Estado/estadio/progreso |
| GET | `/api/v1/download/{job_id}` | Descarga ZIP del run |
| PUT | `/api/v1/datasets/mango/sources/nasa` | Registra la fuente climática NASA POWER (multipart o ruta) (201) |
| PUT | `/api/v1/datasets/mango/sources/local` | Registra los registros de cosecha/pérdida del productor (201) |
| POST | `/api/v1/datasets/mango/sync` | Construye/publica el dataset canónico de mango (idempotente; admite `force`, `dry_run`, `refresh_nasa`) |
| GET | `/api/v1/datasets/mango/manifest` | Manifiesto publicado: cobertura, targets, hashes y warnings |
| GET | `/api/v1/datasets/mango/sources` | Fuentes configuradas, ganadoras y candidatas detectadas |
| GET | `/api/v1/datasets/mango/state` | Resumen de la última sincronización |

## Arranque rápido (Docker)

Único requisito: Docker (Desktop) con Compose v2 en ejecución.

```powershell
# Windows
.\run.ps1
```

```bash
# Linux / macOS (chmod +x run.sh la primera vez)
./run.sh
```

El script, sin configuracopión manual previa: crea `.env` desde `.env.example` con una
`REDIS_PASSWORD` aleatoria si falta (nunca sobrescribe una existente), valida que el motor Docker
esté activo, construye y levanta `api`, `worker`, `beat` y `redis`, y espera hasta que el
healthcheck responda en `http://localhost:8000/api/v1/health`.

| Flag | Efecto |
|---|---|
| `-Test` / `--test` | Ejecuta antes la suite completa (241 tests) en un contenedor efímero y aborta el despliegue si falla |
| `-SkipBuild` / `--skip-build` | Arranca reutilizando imágenes ya construidas |
| `-Down` / `--down` | Detiene el stack |

```powershell
.\run.ps1 -Test   # validar suite en contenedor + desplegar
.\run.ps1 -Down   # parar todo (o: docker compose down)
```

El worker incluye Tectonic para compilar el PDF; sin motor LaTeX el paquete se entrega igualmente
con las fuentes `.tex`. El servicio `beat` programa la tarea `maintenance.cleanup` (limpieza TTL de
`DATA_DIR`). Si la API no responde en 120 s, los scripts vuelcan diagnóstico automático (`ps` +
últimos logs).

## Arranque local (sin Docker)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --reload
```

Sin Redis el sistema opera en modo eager con job store en memoria (útil para desarrollo y CI).

## Frontend (web/)

Interfaz Next.js 16 (App Router, Tailwind, shadcn/ui) con asistente de entrenamiento
(subida de dataset → configuración → lanzamiento), seguimiento de jobs en tiempo real,
dashboard de resultados (métricas por fold con IC95%, comparativa de modelos, galería SHAP,
citación BibTeX) y consola de inferencia (registros manuales o archivo CSV/XLSX con export CSV).
Internacionalización ES/EN con selector en la cabecera.

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

Las peticiones a `/api/*` se proxean al backend vía rewrites (`API_ORIGIN`, por defecto
`http://localhost:8000`); no se necesita CORS. Con el stack Docker levantado basta abrir
el frontend y la aplicación funciona de punta a punta.

## Ejemplo de uso

```bash
curl -F "file=@tests/fixtures/data/mixed_targets.csv" http://localhost:8000/api/v1/datasets/upload

curl -X POST http://localhost:8000/api/v1/train -H "Content-Type: application/json" -d '{
  "dataset_id": "<ds_id>",
  "targets": [
    {"name": "score_continuous", "task": "regression"},
    {"name": "passed_threshold", "task": "classification"}
  ],
  "cv": {"n_folds": 5, "mode": "nested"},
  "hpo": {"enabled": true, "n_trials": 25}
}'
```

## Estructura del ZIP entregado

```
report/report.pdf|report.tex     latex/tables/table_<target>.tex
bib/references.bib               figures/<target>/*.png|eps (+ shap/)
models/<target>_best_model.joblib  models/temporal_context.json (runs temporales)
metrics/results_summary.json     metrics/per_fold.csv
data/dataset_with_predictions.csv  MANIFEST.txt
```

## Inferencia

La respuesta va alineada por fila (`predictions[i].row_index`), con etiquetas decodificadas y
bloques `<target>_probabilities` por clase en clasificación. Columnas extra se ignoran con un
warning. Hay dos formas de enviar datos:

**Opción A — features precalculadas (retrocompatible).** Si el payload ya trae todas las
`feature_columns` del run, incluidas las columnas `*_lag_*` y `*_rolling_*`, se usan tal cual.

```bash
# Registros JSON (tiempo real o lotes pequeños)
curl -X POST "http://localhost:8000/api/v1/predict/<job_id>" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"x1": 0.5, "x2": 5.0, "category": "A"}]}'

# Lote por archivo (CSV/XLSX); ?targets= filtra un subconjunto
curl -X POST "http://localhost:8000/api/v1/predict/<job_id>?targets=grade" \
  -F "file=@nuevos_datos.csv"
```

**Opción B — variables crudas + historial.** Para runs con `temporal`, envía solo las variables
crudas del día (por ejemplo `fecha` y clima). El job persiste el `max_lookback` y las últimas filas
del entrenamiento; la API concatena ese contexto, ordena por fecha y aplica exactamente las mismas
transformaciones causales de `automl_shared.temporal`. Puedes reemplazar el contexto guardado con
`history` (lista de filas crudas anteriores).

```bash
# 3 días futuros usando el historial guardado del entrenamiento
curl -X POST "http://localhost:8000/api/v1/predict/<job_id>" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"fecha": "2026-01-10", "temp_media_c": 27.1, "precipitacion_mm": 0.0},
      {"fecha": "2026-01-11", "temp_media_c": 26.4, "precipitacion_mm": 4.2},
      {"fecha": "2026-01-12", "temp_media_c": 28.0, "precipitacion_mm": 1.1}
    ],
    "horizon": 3,
    "mode": "recursive"
  }'
```

Con `target_lags` activo y varios días sin targets reales intermedios, `mode: "recursive"` alimenta
la predicción de `t` como `target_lag_1` de `t+1`. `mode: "auto"` (por defecto) activa esa recursión
cuando detecta targets intermedios ausentes; `mode: "one-step-ahead"` solo usa valores reales
disponibles y deja NaN los rezagos que no se pueden reconstruir. Si el payload ya trae todas las
features, la recursión no se aplica (se respeta lo precalculado).

Para subir un archivo con historial explícito usa los campos multipart `history` (JSON) y
`horizon`/`mode` (query o formulario).

## Mantenimiento (TTL cleanup)

`maintenance.cleanup` borra `data/jobs/*` y `data/uploads/*` cuya última modificación exceda la
retención (`JOB_RETENTION_HOURS` / `UPLOAD_RETENTION_HOURS`, 168 h por defecto). Los jobs en
estado PENDING/STARTED/PROGRESS nunca se eliminan. Se ejecuta vía Celery Beat cada
`CLEANUP_INTERVAL_MINUTES`; desactívable con `CLEANUP_ENABLED=false`.

## Seguridad

Opt-in por variables de entorno: `AUTH_ENABLED=true` + `API_KEYS=k1,k2` exige `X-API-Key` en las
mutaciones (upload/train/predict; lecturas públicas, comparación constante-tiempo).
`RATE_LIMIT_ENABLED=true`, `RATE_LIMIT_MAX_REQUESTS` y `RATE_LIMIT_WINDOW_SECONDS` activan un
limitador en-proceso (ventana deslizante por API key o IP) que responde 429 + `Retry-After`.

Controles defensivos siempre activos:

- **Nombres de archivo saneados** y **validación estricta de IDs** (`^[A-Za-z0-9_-]{1,64}$`) en
  todas las rutas , sin traversal en `DATA_DIR`.
- **Lectura de uploads por chunks con aborto temprano** (413) y límites estructurales:
  `MAX_DATASET_ROWS`, `MAX_DATASET_COLUMNS`, `PREDICT_MAX_RECORDS`, `PREDICT_MAX_BODY_MB`.
- **Integridad de artefactos**: packaging escribe `models/models.sha256` (HMAC-SHA256 opcional vía
  `ARTIFACT_SIGNING_KEY`); la inferencia verifica cada `.joblib` antes de deserializarlo
  (`ARTIFACT_INTEGRITY=off|if_present|strict`, por defecto `if_present`).
- **Cabeceras HTTP** en `/api/*`: nosniff, X-Frame-Options DENY, Referrer-Policy, CSP
  `default-src 'none'`; HSTS opcional (`HSTS_ENABLED`). **CORS** solo para orígenes listados en
  `CORS_ORIGINS` (por defecto desactivado).
- **Contenedores**: non-root, `no-new-privileges`, `cap_drop ALL`, límites de memoria/CPU, Redis
  con `--requirepass`, red dedicada y `beat` sin volumen de datos.

## Modelos soportados

sklearn core: Linear/Ridge/Lasso, RandomForest, GradientBoosting · XGBoost · LightGBM.
Selección automática por target; métricas primarias: `f1_macro` (clasificación), `rmse` (regresión).

## Datos temporales y motor tabular / Temporal data and the tabular engine

**ES**

**Rezagos y ventanas causales.** El bloque `temporal` del request activa ingeniería de
características temporales estrictamente causal para cualquier CSV/XLSX con columna de fecha y para
el dataset gestionado de mango. No hay nombres de columnas hardcodeados: las columnas exógenas y los
targets se pasan por configuración, así que funciona igual con `PESO (PRODUCCIÓN)`,
`Peso_Recolectado_KG` o `Peso_Perdido_KG`.

- `lags` (por defecto `[1, 3, 7, 14, 30]`) genera `{columna}_lag_{k}` con `shift(k)` sobre las
  columnas exógenas seleccionadas.
- `rolling_windows` (por defecto `[7, 14, 30]`) y `rolling_aggs` (`sum`, `mean`, `max`, `min`)
  generan `{columna}_{agg}_{w}d`. Con `include_current_day: true` la ventana es `<= t`; con
  `false` se desplaza a `t-w .. t-1`, útil para inferencia operativa antes de tener el clima del día.
- `target_lags` (p. ej. `[1, 2, 3, 7]`) genera `{target}_lag_{k}` sobre los targets configurados,
  ordenando previamente por fecha. Es la autocorrelación de cosecha.
- `drop_zero_targets` (por defecto `false`) elimina las filas donde **todos** los targets configurados
  valen `0`, después de calcular lags y ventanas sobre la serie diaria completa. Así se evita el
  sesgo de días sin corte sin romper la continuidad temporal.

**Guardarraíl anti-fuga:** si `target_lags` no está vacío, `cv.scheme` debe ser `"timeseries"` y
`cv.gap >= max(target_lags)`. En caso contrario el request se rechaza con `ValidationFailedError`.
Así ningún valor del bloque de validación puede entrar en las features de entrenamiento, y los
splits internos de `nested`/Optuna respetan el mismo `gap`.

```json
{
  "cv": {"scheme": "timeseries", "gap": 7, "date_column": "fecha"},
  "temporal": {
    "enabled": true,
    "exogenous_columns": ["temp_media_c", "precipitacion_mm"],
    "lags": [1, 3, 7, 14, 30],
    "rolling_windows": [7, 14, 30],
    "rolling_aggs": ["sum", "mean", "max", "min"],
    "include_current_day": true,
    "target_columns": ["PESO (PRODUCCIÓN)", "Peso_Recolectado_KG"],
    "target_lags": [1, 2, 3, 7]
  }
}
```

Si `enabled` es `true` y `exogenous_columns` queda vacío, se autoseleccionan las columnas numéricas
que no son targets ni la fecha. Si `target_columns` queda vacío, se usan los targets del request. El
run persiste la configuración resuelta, `max_lookback = max(lags + rolling_windows + target_lags)` y
las últimas `max_lookback` filas crudas del entrenamiento. La inferencia puede recibir las
`feature_columns` ya calculadas (Opción A) o solo las variables crudas del día (Opción B): en este
segundo caso reconstruye las features con el historial guardado o con `history` explícito.

> **Validación temporal:** por defecto la validación cruzada usa `KFold` con mezcla aleatoria de
> filas (`services/ml-core/mlcore/ml/validation/kfold.py`). En series de tiempo esto puede producir
> métricas optimistas por autocorrelación, porque días vecinos caen a la vez en train y test. Para
> conclusiones temporales usa `"cv": {"scheme": "timeseries", "gap": N}`: el pipeline ordena por la
> columna de fecha (autodetectada o indicada en `date_column`), entrena con ventana expansiva y deja
> `N` filas de separación entre train y test.

**EN**

**Causal lags and rolling windows.** The request's `temporal` block enables strictly causal temporal
feature engineering for any CSV/XLSX with a date column, as well as for the managed mango dataset.
No column names are hardcoded: exogenous columns and targets are supplied through configuration, so
it works the same with `PESO (PRODUCCIÓN)`, `Peso_Recolectado_KG` or `Peso_Perdido_KG`.

- `lags` (default `[1, 3, 7, 14, 30]`) creates `{column}_lag_{k}` with `shift(k)` on the selected
  exogenous columns.
- `rolling_windows` (default `[7, 14, 30]`) and `rolling_aggs` (`sum`, `mean`, `max`, `min`) create
  `{column}_{agg}_{w}d`. With `include_current_day: true` the window is `<= t`; with `false` it is
  shifted to `t-w .. t-1`, useful for operational inference before the current day's weather exists.
- `target_lags` (e.g. `[1, 2, 3, 7]`) creates `{target}_lag_{k}` on the configured targets after
  sorting by date. It captures harvest autocorrelation.
- `drop_zero_targets` (default `false`) removes rows where **every** configured target is `0`, after
  computing lags and rolling windows over the complete daily series. This reduces zero-inflation
  without breaking temporal continuity.

**Anti-leakage guardrail:** when `target_lags` is not empty, `cv.scheme` must be `"timeseries"` and
`cv.gap >= max(target_lags)`. Otherwise the request is rejected with `ValidationFailedError`. This
keeps every validation-block value out of the training features, and `nested`/Optuna inner splits
honor the same `gap`.

```json
{
  "cv": {"scheme": "timeseries", "gap": 7, "date_column": "fecha"},
  "temporal": {
    "enabled": true,
    "exogenous_columns": ["temp_media_c", "precipitacion_mm"],
    "lags": [1, 3, 7, 14, 30],
    "rolling_windows": [7, 14, 30],
    "rolling_aggs": ["sum", "mean", "max", "min"],
    "include_current_day": true,
    "target_columns": ["PESO (PRODUCCIÓN)", "Peso_Recolectado_KG"],
    "target_lags": [1, 2, 3, 7]
  }
}
```

When `enabled` is `true` and `exogenous_columns` is empty, numeric non-target columns other than the
date are auto-selected. When `target_columns` is empty, the request's targets are used. The run
persists the resolved configuration, `max_lookback = max(lags + rolling_windows + target_lags)` and
the last `max_lookback` raw training rows. Inference accepts the run's precomputed
`feature_columns` (Option A) or just the day's raw covariates (Option B); in the latter case it
rebuilds the features from the saved history or an explicit `history` payload. With `target_lags`
and multiple future days, use `mode: "recursive"` (or let `auto` detect missing intermediate
targets).

> **Temporal validation:** by default cross-validation uses `KFold` with random row shuffling
> (`services/ml-core/mlcore/ml/validation/kfold.py`). On time series this can yield optimistic
> metrics due to autocorrelation, because neighboring days land in both train and test. For temporal
> conclusions use `"cv": {"scheme": "timeseries", "gap": N}`: the pipeline sorts by the date column
> (auto-detected or set via `date_column`), trains on an expanding window and leaves `N` rows between
> train and test.

## Tests

```bash
# Tier 1: suite completa contra el monolito componible (rápido, eager)
.venv\Scripts\python -m pytest tests -q -p no:warnings

# Regenerar fixtures E2E
python tests/fixtures/make_fixtures.py
```

```bash
# Tier 2: flujo de humo contra el stack de microservicios real
docker compose --profile tier2 run --rm tier2
```

## Contribuir / Contributing

¡Las contribuciones son bienvenidas! Lee la [Guía de Contribución](CONTRIBUTING.md)
para configurar tu entorno y enviar cambios.

Contributions are welcome! Read the [Contributing Guide](CONTRIBUTING.md) to set
up your environment and submit changes.

## Seguridad / Security

Para reportar vulnerabilidades de seguridad, envía un email a
**juanc-llamasc@qbitsglobal.com**. No abras issues públicas.
Ver [SECURITY.md](SECURITY.md) para más detalles.

To report security vulnerabilities, email **juanc-llamasc@qbitsglobal.com**.
Do not open public issues. See [SECURITY.md](SECURITY.md) for details.

## Código de Conducta / Code of Conduct

Este proyecto sigue el [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Licencia / License

Este proyecto está licenciado bajo la [MIT License](LICENSE).

This project is licensed under the [MIT License](LICENSE).

## Citación de software / Software Citation

`bib/references.bib` incluye entradas BibTeX curadas (scikit-learn, XGBoost, LightGBM, Optuna, SHAP, NumPy, pandas, matplotlib, seaborn) con versiones exactas registradas en cada run.

`bib/references.bib` includes curated BibTeX entries (scikit-learn, XGBoost, LightGBM, Optuna, SHAP, NumPy, pandas, matplotlib, seaborn) with exact versions recorded in each run.
