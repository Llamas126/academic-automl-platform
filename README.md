# academic-automl-platform

Backend asíncrono de AutoML para investigación académica: ingesta de tablas CSV/XLSX, modelado multi-target (regresión y clasificación mixtas), comparación de algoritmos con validación cruzada anidada, optimización de hiperparámetros con Optuna, explicabilidad SHAP y entrega de un paquete ZIP listo para publicación (reporte PDF/LaTeX, tablas booktabs, BibTeX, figuras 300 DPI PNG+EPS, modelos `.joblib` y predicciones out-of-fold).

## Arquitectura

- **API**: FastAPI (`app/main.py`), rutas en `app/api/v1/`
- **Tareas**: Celery + Redis (`app/tasks/`); fallback a ejecución eager + job store en memoria cuando Redis no está disponible
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
| GET | `/api/v1/status/{job_id}` | Estado/estadio/progreso |
| GET | `/api/v1/download/{job_id}` | Descarga ZIP del run |

## Arranque rápido (Docker)

```bash
docker compose up --build
# API en http://localhost:8000 — docs en /docs
```

El worker incluye Tectonic para compilar el PDF; sin motor LaTeX el paquete se entrega igualmente con las fuentes `.tex`.

## Arranque local (sin Docker)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --reload
```

Sin Redis el sistema opera en modo eager con job store en memoria (útil para desarrollo y CI).

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
