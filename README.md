# academic-automl-platform

Backend asíncrono de AutoML para investigación académica: ingesta de tablas CSV/XLSX, modelado multi-target (regresión y clasificación mixtas), comparación de algoritmos con validación cruzada anidada, optimización de hiperparámetros con Optuna, explicabilidad SHAP y entrega de un paquete ZIP listo para publicación (reporte PDF/LaTeX, tablas booktabs, BibTeX, figuras 300 DPI PNG+EPS, modelos `.joblib` y predicciones out-of-fold).

## Arquitectura

- **API**: FastAPI (`app/main.py`), rutas en `app/api/v1/`
- **Tareas**: Celery + Redis (`app/tasks/`); fallback a ejecución eager + job store en memoria cuando Redis no está disponible; beat programado para limpieza TTL de artefactos
- **Seguridad**: middleware ligero opt-in (`app/core/security.py`): API key para mutaciones y rate limit en-proceso
- **Núcleo ML**: `app/ml/` (profiling, preprocesado, registry de modelos, K-Fold compartido, métricas IC95%, Optuna, multi-target, pipeline)
- **Exportación científica**: `app/services/` (figuras, LaTeX booktabs, BibTeX, reporte Jinja→Tectonic, empaquetado ZIP+MANIFEST)

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/v1/health` | Liveness |
| POST | `/api/v1/datasets/analyze-dataset` | Sube CSV/XLSX, devuelve profiling y sugerencias de targets (201) |
| GET | `/api/v1/datasets/{dataset_id}` | Metadatos del dataset |
| DELETE | `/api/v1/datasets/{dataset_id}` | Borra dataset |
| POST | `/api/v1/train` | Crea job de entrenamiento (202) |
| POST | `/api/v1/predict/{job_id}` | Inferencia con los modelos de un run terminado: JSON `{"records":[...]}` o upload CSV/XLSX; query opcional `?targets=t1,t2` |
| GET | `/api/v1/status/{job_id}` | Estado/estadio/progreso |
| GET | `/api/v1/download/{job_id}` | Descarga ZIP del run |

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

El script, sin configuración manual previa: crea `.env` desde `.env.example` con una
`REDIS_PASSWORD` aleatoria si falta (nunca sobrescribe una existente), valida que el motor Docker
esté activo, construye y levanta `api`, `worker`, `beat` y `redis`, y espera hasta que el
healthcheck responda en `http://localhost:8000/api/v1/health`.

| Flag | Efecto |
|---|---|
| `-Test` / `--test` | Ejecuta antes la suite completa (112 tests) en un contenedor efímero y aborta el despliegue si falla |
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
curl -F "file=@tests/fixtures/data/mixed_targets.csv" http://localhost:8000/api/v1/datasets/analyze-dataset

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
models/<target>_best_model.joblib
metrics/results_summary.json     metrics/per_fold.csv
data/dataset_with_predictions.csv  MANIFEST.txt
```

## Inferencia

```bash
# Registros JSON (tiempo real o lotes pequeños)
curl -X POST "http://localhost:8000/api/v1/predict/<job_id>" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"x1": 0.5, "x2": 5.0, "category": "A"}]}'

# Lote por archivo (CSV/XLSX); ?targets= filtra un subconjunto
curl -X POST "http://localhost:8000/api/v1/predict/<job_id>?targets=grade" \
  -F "file=@nuevos_datos.csv"
```

La respuesta va alineada por fila (`predictions[i].row_index`), con etiquetas decodificadas y
bloques `<target>_probabilities` por clase en clasificación. Columnas extra se ignoran con un
warning; columnas faltantes devuelven 422.

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
  todas las rutas — sin traversal en `DATA_DIR`.
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

## Tests

```bash
.venv\Scripts\python -m pytest tests -q
python tests/fixtures/make_fixtures.py   # regenera fixtures E2E
```

## Citación de software

`bib/references.bib` incluye entradas BibTeX curadas (scikit-learn, XGBoost, LightGBM, Optuna, SHAP, NumPy, pandas, matplotlib, seaborn) con versiones exactas registradas en cada run.
