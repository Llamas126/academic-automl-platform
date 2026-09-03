# Flujo Completo del Software

Este documento describe paso a paso todo lo que ocurre cuando un usuario utiliza la plataforma.

---

## Diagrama de Flujo General

```
┌──────────────────────────────────────────────────────────────────┐
│                        USUARIO                                   │
│   1. Sube archivo CSV/Excel                                      │
│   2. Selecciona variable objetivo                                │
│   3. Elige modelos y configuracion                               │
│   4. Hace clic en "Entrenar"                                     │
└──────────────┬───────────────────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│  FASE 1: INGESTION (recepcion de datos)                         │
│                                                                  │
│  • Valida tipo de archivo (CSV, XLSX, XLS)                      │
│  • Valida tamano (maximo 200 MB)                                │
│  • Detecta separador (coma, punto y coma, tab, tuberia)         │
│  • Valida: no vacio, minimo 2 columnas, max 200K filas          │
│  • Analiza cada columna (tipo, nulos, unicos)                   │
│  • Sugiere variables objetivo                                    │
└──────────────┬───────────────────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│  FASE 2: ENTRENAMIENTO (la mas larga)                           │
│                                                                  │
│  2a. Preparacion de datos:                                       │
│      • Detectar tipo de tarea (regresion o clasificacion)       │
│      • Eliminar filas con target NaN                             │
│      • Codificar labels (LabelEncoder para clasificacion)        │
│      • Identificar columnas predictoras                          │
│                                                                  │
│  2b. Preprocesamiento:                                           │
│      • Imputar nulos (mediana numerico, moda categorico)         │
│      • Escalar si el modelo lo necesita (Ridge, Lasso, LogReg)  │
│      • One-Hot Encoding para categorias                          │
│                                                                  │
│  2c. Para cada modelo:                                           │
│      • Dividir en K folds (validacion cruzada)                  │
│      • Para cada fold:                                           │
│          - Optimizar hiperparametros (Optuna, TPE)              │
│          - Entrenar modelo con mejores hiperparametros           │
│          - Evaluar en test del fold                              │
│          - Guardar predicciones out-of-fold                      │
│      • Promediar metricas across folds                           │
│      • Calcular intervalo de confianza al 95%                    │
│                                                                  │
│  2d. Seleccion del mejor modelo:                                 │
│      • Rankear por metrica principal (RMSE o F1)                │
│      • Re-entrenar el ganador con TODOS los datos               │
│      • Guardar modelo (.joblib) + manifest SHA-256              │
└──────────────┬───────────────────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│  FASE 3: EXPLICABILIDAD (SHAP)                                  │
│                                                                  │
│  • Seleccionar estrategia segun tipo de modelo                  │
│  • Calcular valores SHAP para cada prediccion                   │
│  • Generar graficos:                                             │
│      - Bar plot (importancia global)                             │
│      - Beeswarm (resumen global + direccion)                    │
│      - Dependence plot (relacion de feature top)                 │
│  • Guardar graficos PNG a 300 DPI                                │
└──────────────┬───────────────────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│  FASE 4: EXPORTACION (generacion de informe)                    │
│                                                                  │
│  • Generar graficos diagnosticos:                                │
│      - Matriz de confusion (clasificacion)                      │
│      - Curvas ROC (clasificacion)                                │
│      - Scatter predicho vs real (regresion)                     │
│      - Plot de residuos (regresion)                              │
│      - Boxplot de metricas por fold                              │
│      - Heatmap de correlacion                                    │
│  • Generar tablas LaTeX con metricas (mean +/- std)             │
│  • Generar documento LaTeX completo (booktabs, amsmath)         │
│  • Compilar a PDF con Tectonic                                   │
│  • Generar citas BibTeX automaticas                              │
│  • Empaquetar todo en ZIP:                                       │
│      - Modelo .joblib                                            │
│      - Informe PDF                                               │
│      - Figuras PNG                                               │
│      - Datos out-of-fold                                         │
│      - Citas BibTeX                                              │
└──────────────┬───────────────────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────────────────┐
│  FASE 5: PREDICCION (opcional, despues del entrenamiento)       │
│                                                                  │
│  • Cargar modelo guardado (.joblib)                             │
│  • Verificar integridad (SHA-256)                                │
│  • Recibir nuevos datos (JSON o CSV)                            │
│  • Aplicar mismo preprocesamiento                                │
│  • Ejecutar pipeline.predict()                                  │
│  • Para clasificacion: tambien predict_proba()                   │
│  • Devolver predicciones al usuario                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Tiempos tipicos

| Fase | Tiempo estimado | Que influye |
|------|----------------|-------------|
| Ingestion | 5-30 segundos | Tamano del archivo |
| Preparacion | 1-5 segundos | Numero de columnas |
| Entrenamiento | 5-60 minutos | Modelos, HPO trials, folds, tamano del dataset |
| SHAP | 1-10 minutos | Tamano del dataset, tipo de modelo |
| Exportacion | 1-5 minutos | Numero de targets, graficos |
| **Total** | **7-75 minutos** | Depende de la complejidad |

---

## Arquitectura de microservicios

El software esta dividido en 6 servicios independientes:

```
                    ┌─────────────┐
                    │   GATEWAY   │  Puerto 8000
                    │ (puerta de  │  Autenticacion + rate limiting
                    │  entrada)   │  + reverse proxy
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            v              v              v
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │INGESTION │   │ ML-CORE  │   │INFERENCE │
    │          │   │          │   │          │
    │ Subir y  │   │ Entrenar │   │ Predecir │
    │ analizar │   │ modelos  │   │ con modelo│
    │ datos    │   │          │   │ guardado │
    └──────────┘   └────┬─────┘   └──────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              v                   v
      ┌──────────────┐   ┌──────────────┐
      │  EXPLAIN     │   │   EXPORTER   │
      │  (SHAP)      │   │ (informes)   │
      │              │   │              │
      │ Generar      │   │ Generar PDF, │
      │ graficos de  │   │ LaTeX, ZIP,  │
      │ explicacion  │   │ BibTeX       │
      └──────────────┘   └──────────────┘
```

### Comunicacion entre servicios

- **HTTP:** El Gateway reenvia peticiones al servicio correcto
- **Celery + Redis:** Las tareas pesadas (entrenar, SHAP, exportar) se ejecutan en segundo plano
- **Archivos compartidos:** Los servicios comparten datos a traves del filesystem (`data/jobs/`)

### Cadena de Celery

Cuando el usuario hace clic en "Entrenar", se lanza una **cadena asincrona**:

```
train.run  -->  explain.run  -->  export.package
```

Cada paso solo empieza cuando el anterior termina. Si algun paso falla, la cadena se detiene y el usuario ve el error.

---

## Interfaz de usuario

La interfaz tiene 4 paginas principales:

### 1. Pagina de inicio: Asistente de entrenamiento (wizard de 3 pasos)

```
Paso 1: Subir Dataset
  └── Drag & drop de CSV/Excel
  └── Analisis automatico de columnas
  └── Sugerencia de variable objetivo

Paso 2: Configuracion
  └── Seleccionar target y tarea
  └── Elegir modelos (checkboxes)
  └── Configurar validacion cruzada
  └── Activar/desactivar HPO
  └── Activar/desactivar SHAP

Paso 3: Revision
  └── Resumen de toda la configuracion
  └── Boton "Entrenar"
```

### 2. Pagina de estado del trabajo

- Barra de progreso en tiempo real (cada 2 segundos)
- Etapa actual (entrenando, explicando, exportando)
- Tiempo transcurrido
- Pasos completados (stepper visual)

### 3. Pagina de resultados

- Tarjetas resumen (muestras, features, tiempo)
- Pestanas por variable objetivo
- Grafico de barras comparando modelos
- Tabla detallada de metricas con IC 95%
- Galeria de graficos SHAP
- Cita BibTeX listo para copiar
- Boton para descargar ZIP

### 4. Pagina de prediccion

- Ingresar ID de trabajo entrenado
- Modo manual: tabla editable fila por fila
- Modo archivo: subir CSV con nuevos datos
- Resultado: tabla de predicciones con probabilidades
- Exportar predicciones a CSV

---

## Datos que almacena el software

```
data/
├── uploads/                    # Datasets subidos por el usuario
│   └── ds_{uuid}/
│       ├── archivo.csv         # El archivo original
│       └── profile.json        # Analisis de columnas
│
└── jobs/                       # Trabajos de entrenamiento
    └── {job_id}/
        ├── work/               # Datos intermedios (contrato entre servicios)
        │   └── training_contract.json
        │
        └── output/             # Resultados finales
            ├── models/
            │   └── *_best_model.joblib
            ├── metrics/
            │   └── results.json
            ├── figures/
            │   └── *.png
            └── data/
                └── oof_predictions.csv
            
            results_{job_id}.zip    # Paquete para descargar
```
