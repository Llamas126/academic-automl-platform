"use client";

import { createContext, useCallback, useContext, useMemo, useSyncExternalStore } from "react";

export type Lang = "es" | "en";

const es: Record<string, string> = {
  "app.title": "Academic AutoML",
  "app.subtitle": "Plataforma de AutoML para investigación",
  "nav.train": "Entrenar",
  "nav.predict": "Predecir",
  "nav.docs": "API Docs",
  "lang.label": "Idioma",

  "wizard.step1": "Dataset",
  "wizard.step2": "Configuración",
  "wizard.step3": "Revisión",
  "wizard.next": "Siguiente",
  "wizard.back": "Atrás",
  "wizard.launch": "Lanzar entrenamiento",

  "upload.title": "Sube tu dataset",
  "upload.hint": "Arrastra un archivo CSV o XLSX aquí, o haz clic para seleccionarlo",
  "upload.maxSize": "Tamaño máximo: {size} MB",
  "upload.uploading": "Subiendo… {pct}%",
  "upload.success": "Dataset cargado y perfilado correctamente",
  "upload.failed": "Error al subir el dataset",

  "profile.rows": "Filas",
  "profile.columns": "Columnas",
  "profile.file": "Archivo",
  "profile.colName": "Columna",
  "profile.colRole": "Tipo",
  "profile.missing": "% Faltante",
  "profile.unique": "Únicos",
  "profile.suggestions": "Objetivos sugeridos",
  "profile.excluded": "excluida",

  "config.targets": "Variable(s) objetivo",
  "config.targetsHint": "Selecciona una o varias columnas a predecir",
  "config.task": "Tarea",
  "config.task.auto": "Automática",
  "config.task.classification": "Clasificación",
  "config.task.regression": "Regresión",
  "config.models": "Modelos candidatos",
  "config.modelsAll": "Todos los compatibles por tarea",
  "config.advanced": "Opciones avanzadas",
  "config.cv": "Validación cruzada",
  "config.folds": "Pliegues externos",
  "config.cvMode": "Modo CV",
  "config.cvMode.nested": "Anidada (HPO sin fuga)",
  "config.cvMode.single": "Simple",
  "config.innerFolds": "Pliegues internos",
  "config.seed": "Semilla aleatoria",
  "config.hpo": "Optimización de hiperparámetros",
  "config.hpo.enabled": "Activar HPO (Optuna)",
  "config.trials": "Número de pruebas",
  "config.timeout": "Tiempo límite (s)",
  "config.timeoutOff": "sin límite",
  "config.explain": "Explicabilidad",
  "config.shap": "Incluir análisis SHAP",
  "config.background": "Muestras de fondo",
  "config.maxDisplay": "Máx. características mostradas",
  "config.export": "Exportación científica",
  "config.dpi": "Resolución de figuras (DPI)",
  "config.eps": "Incluir vectorial EPS",
  "config.dependence": "Gráficos de dependencia SHAP",

  "review.title": "Resumen del experimento",
  "review.dataset": "Dataset",
  "review.targetsList": "Objetivos",
  "review.selectedModels": "Modelos",
  "review.readyToLaunch": "Todo listo para lanzar",

  "jobs.title": "Entrenamiento",
  "jobs.notFound": "Job no encontrado",
  "jobs.stage": "Etapa actual",
  "jobs.progress": "Progreso",
  "jobs.elapsed": "Tiempo transcurrido",
  "jobs.state.PENDING": "En cola",
  "jobs.state.STARTED": "Iniciado",
  "jobs.state.PROGRESS": "Entrenando",
  "jobs.state.SUCCESS": "Completado",
  "jobs.state.FAILURE": "Fallido",
  "jobs.failedDetail": "El entrenamiento terminó con error",
  "jobs.viewResults": "Ver resultados",
  "jobs.downloadZip": "Descargar ZIP",
  "jobs.newTrain": "Nuevo entrenamiento",

  "results.title": "Resultados",
  "results.dataset": "Dataset",
  "results.samples": "Muestras",
  "results.features": "Características",
  "results.runtime": "Duración",
  "results.tab.overview": "Comparativa",
  "results.tab.shap": "SHAP",
  "results.bestModel": "Mejor modelo",
  "results.bestParams": "Mejores hiperparámetros",
  "results.rank": "Puesto",
  "results.model": "Modelo",
  "results.metric": "Métrica principal",
  "results.allMetrics": "Todas las métricas",
  "results.ci95": "IC 95%",
  "results.showFolds": "Ver pliegues",
  "results.warnings": "Advertencias",
  "results.shapEmpty": "No hay figuras SHAP para este objetivo (análisis desactivado o no aplicable)",
  "results.shapHint": "Haz clic en una figura para ampliarla",
  "results.cite": "Citar este resultado",
  "results.copyBibtex": "Copiar BibTeX",
  "results.copied": "Copiado al portapapeles",
  "results.predictCta": "Predecir con este modelo",
  "results.downloadZip": "Descargar paquete ZIP",
  "results.downloading": "Descargando…",

  "predict.title": "Inferencia",
  "predict.jobId": "ID del job entrenado",
  "predict.load": "Cargar modelo",
  "predict.loadingModel": "Cargando artefactos…",
  "predict.mode.records": "Registro por registro",
  "predict.mode.file": "Archivo batch (CSV/XLSX)",
  "predict.addRow": "Añadir fila",
  "predict.removeRow": "Eliminar",
  "predict.runRecords": "Predecir registros",
  "predict.dropFile": "Arrastra un CSV/XLSX con las columnas de entrenamiento",
  "predict.runFile": "Predecir archivo",
  "predict.resultsFor": "Predicciones ({n} filas)",
  "predict.downloadCsv": "Exportar CSV",
  "predict.probabilities": "probabilidades",
  "predictions.target": "Predicción",

  "common.loading": "Cargando…",
  "common.error": "Error",
  "common.retry": "Reintentar",
  "common.close": "Cerrar",
  "common.of": "de",
};

const en: Record<string, string> = {
  "app.title": "Academic AutoML",
  "app.subtitle": "AutoML platform for research",
  "nav.train": "Train",
  "nav.predict": "Predict",
  "nav.docs": "API Docs",
  "lang.label": "Language",

  "wizard.step1": "Dataset",
  "wizard.step2": "Configuration",
  "wizard.step3": "Review",
  "wizard.next": "Next",
  "wizard.back": "Back",
  "wizard.launch": "Launch training",

  "upload.title": "Upload your dataset",
  "upload.hint": "Drag a CSV or XLSX file here, or click to browse",
  "upload.maxSize": "Maximum size: {size} MB",
  "upload.uploading": "Uploading… {pct}%",
  "upload.success": "Dataset uploaded and profiled successfully",
  "upload.failed": "Failed to upload dataset",

  "profile.rows": "Rows",
  "profile.columns": "Columns",
  "profile.file": "File",
  "profile.colName": "Column",
  "profile.colRole": "Type",
  "profile.missing": "Missing %",
  "profile.unique": "Unique",
  "profile.suggestions": "Suggested targets",
  "profile.excluded": "excluded",

  "config.targets": "Target variable(s)",
  "config.targetsHint": "Select one or more columns to predict",
  "config.task": "Task",
  "config.task.auto": "Automatic",
  "config.task.classification": "Classification",
  "config.task.regression": "Regression",
  "config.models": "Candidate models",
  "config.modelsAll": "All compatible per task",
  "config.advanced": "Advanced options",
  "config.cv": "Cross-validation",
  "config.folds": "Outer folds",
  "config.cvMode": "CV mode",
  "config.cvMode.nested": "Nested (leak-free HPO)",
  "config.cvMode.single": "Single",
  "config.innerFolds": "Inner folds",
  "config.seed": "Random seed",
  "config.hpo": "Hyperparameter optimization",
  "config.hpo.enabled": "Enable HPO (Optuna)",
  "config.trials": "Number of trials",
  "config.timeout": "Timeout (s)",
  "config.timeoutOff": "no limit",
  "config.explain": "Explainability",
  "config.shap": "Include SHAP analysis",
  "config.background": "Background samples",
  "config.maxDisplay": "Max displayed features",
  "config.export": "Scientific export",
  "config.dpi": "Figure resolution (DPI)",
  "config.eps": "Include EPS vector graphics",
  "config.dependence": "SHAP dependence plots",

  "review.title": "Experiment summary",
  "review.dataset": "Dataset",
  "review.targetsList": "Targets",
  "review.selectedModels": "Models",
  "review.readyToLaunch": "Ready to launch",

  "jobs.title": "Training",
  "jobs.notFound": "Job not found",
  "jobs.stage": "Current stage",
  "jobs.progress": "Progress",
  "jobs.elapsed": "Elapsed time",
  "jobs.state.PENDING": "Queued",
  "jobs.state.STARTED": "Started",
  "jobs.state.PROGRESS": "Training",
  "jobs.state.SUCCESS": "Completed",
  "jobs.state.FAILURE": "Failed",
  "jobs.failedDetail": "The training run finished with an error",
  "jobs.viewResults": "View results",
  "jobs.downloadZip": "Download ZIP",
  "jobs.newTrain": "New training run",

  "results.title": "Results",
  "results.dataset": "Dataset",
  "results.samples": "Samples",
  "results.features": "Features",
  "results.runtime": "Runtime",
  "results.tab.overview": "Comparison",
  "results.tab.shap": "SHAP",
  "results.bestModel": "Best model",
  "results.bestParams": "Best hyperparameters",
  "results.rank": "Rank",
  "results.model": "Model",
  "results.metric": "Primary metric",
  "results.allMetrics": "All metrics",
  "results.ci95": "95% CI",
  "results.showFolds": "View folds",
  "results.warnings": "Warnings",
  "results.shapEmpty": "No SHAP figures for this target (analysis disabled or not applicable)",
  "results.shapHint": "Click a figure to enlarge it",
  "results.cite": "Cite this result",
  "results.copyBibtex": "Copy BibTeX",
  "results.copied": "Copied to clipboard",
  "results.predictCta": "Predict with this model",
  "results.downloadZip": "Download ZIP package",
  "results.downloading": "Downloading…",

  "predict.title": "Inference",
  "predict.jobId": "Trained job ID",
  "predict.load": "Load model",
  "predict.loadingModel": "Loading artifacts…",
  "predict.mode.records": "Record by record",
  "predict.mode.file": "Batch file (CSV/XLSX)",
  "predict.addRow": "Add row",
  "predict.removeRow": "Remove",
  "predict.runRecords": "Predict records",
  "predict.dropFile": "Drag a CSV/XLSX file containing the training feature columns",
  "predict.runFile": "Predict file",
  "predict.resultsFor": "Predictions ({n} rows)",
  "predict.downloadCsv": "Export CSV",
  "predict.probabilities": "probabilities",
  "predictions.target": "Prediction",

  "common.loading": "Loading…",
  "common.error": "Error",
  "common.retry": "Retry",
  "common.close": "Close",
  "common.of": "of",
};

const DICTIONARIES: Record<Lang, Record<string, string>> = { es, en };

interface I18nContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

const LANG_KEY = "automl-lang";
const langListeners = new Set<() => void>();
let cachedLang: string | null | undefined;

function subscribeLang(listener: () => void) {
  langListeners.add(listener);
  return () => {
    langListeners.delete(listener);
  };
}

function getClientLang(): string | null {
  if (cachedLang === undefined) {
    cachedLang = window.localStorage.getItem(LANG_KEY);
  }
  return cachedLang;
}

function getServerLang(): string | null {
  return null;
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const stored = useSyncExternalStore(subscribeLang, getClientLang, getServerLang);
  const lang: Lang = stored === "en" ? "en" : "es";

  const setLang = useCallback((next: Lang) => {
    window.localStorage.setItem(LANG_KEY, next);
    document.documentElement.lang = next;
    cachedLang = next;
    for (const listener of langListeners) listener();
  }, []);

  const t = useCallback(
    (key: string, vars?: Record<string, string | number>) => {
      let text = DICTIONARIES[lang][key] ?? DICTIONARIES.en[key] ?? key;
      if (vars) {
        for (const [name, value] of Object.entries(vars)) {
          text = text.replaceAll(`{${name}}`, String(value));
        }
      }
      return text;
    },
    [lang],
  );

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) throw new Error("useI18n must be used within I18nProvider");
  return context;
}
