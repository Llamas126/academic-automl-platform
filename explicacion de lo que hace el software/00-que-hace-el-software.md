# Que hace este software

## Explicacion simple

Imagina que tienes una hoja de Excel con muchos datos numericos y quieres **predecir algo** o **clasificar algo**. Por ejemplo:

- Tienes datos de casas (metros cuadrados, numero de habitaciones, zona) y quieres **predecir el precio**.
- Tienes datos de pacientes (edad, peso, presion arterial) y quieres **clasificar si tienen una enfermedad o no**.

Normalmente, para hacer esto necesitarias:

1. Conocer estadistica avanzada
2. Saber programar en Python
3. Conocer multiples algoritmos de machine learning
4. Saber cual algoritmo funciona mejor para tus datos

**Este software elimina todo eso.** Tu solo subes tu archivo de datos, elegis que queres predecir, y el software hace todo el trabajo automaticamente:

- Prueba 12 algoritmos diferentes
- Encuentra el mejor para tus datos
- Te da metricas de precision
- Te explica porque tomo esa decision
- Genera un informe cientificado listo para entregar

## Flujo en 4 pasos

```
PASO 1: Subir datos (CSV o Excel)
         |
         v
PASO 2: Elegir que predecir (variable objetivo)
         |
         v
PASO 3: El software entrena 12 modelos automaticamente
         |
         v
PASO 4: Recibir resultados + informe + modelo guardado
```

## Que produce el software

Al finalizar el entrenamiento, el usuario recibe un archivo ZIP que contiene:

| Archivo | Que es |
|---------|--------|
| **Modelo entrenado** (.joblib) | El algoritmo guardado, listo para hacer predicciones nuevas |
| **Informe PDF** | Documento cientifico con tablas de metricas, graficos y explicaciones |
| **Archivos BibTeX** | Citas academicas listas para usar en un paper |
| **Figuras PNG** | Graficos de confusion, curvas ROC, graficos SHAP de explicabilidad |
| **Datos out-of-fold** | Las predicciones del modelo en datos que nunca vio durante el entrenamiento |

## Que NO es este software

- **No es un reemplazo del entendimiento.** Sirve para explorar rapidamente, pero el investigador debe interpretar los resultados.
- **No funciona para cualquier cosa.** Requiere datos tabulares (filas y columnas, como una hoja de calculo).
- **No garantiza resultados perfectos.** La calidad del modelo depende de la calidad de los datos.

## Tecnologia detras del software

El software esta construido con 6 servicios que trabajan juntos:

```
Usuario  -->  Gateway (puerta de entrada)
                   |
            Ingestion (recibe y analiza los datos)
                   |
            ML Core (entrena los modelos)
                   |
            SHAP Worker (explica las decisiones)
                   |
            Exporter (genera el informe)
                   |
            Inference (hace predicciones nuevas)
```

Todo esto corre en **contenedores Docker**, que son como cajitas independientes que se comunican entre si.
