# Validacion Cruzada y Optimizacion de Hiperparametros

## Por que no es suficiente entrenar una sola vez?

Si entrenamos el modelo con todos los datos y lo evaluamos con esos mismos datos, el resultado sera **artificialmente bueno**. Es como un estudiante que estudia con el examen en la mano: saca 100%, pero no sabe nada.

Necesitamos **evaluar el modelo con datos que nunca vio**.

---

## Validacion Cruzada K-Fold

### Concepto simple
Se divide los datos en K partes (folds). Se entrena con K-1 partes y se evalua con la parte restante. Se repite K veces, cada vez usando una parte diferente para evaluacion.

### Ejemplo con K=5

```
Iteracion 1: [TEST][Train][Train][Train][Train]  -> Error en test 1
Iteracion 2: [Train][TEST][Train][Train][Train]  -> Error en test 2
Iteracion 3: [Train][Train][TEST][Train][Train]  -> Error en test 3
Iteracion 4: [Train][Train][Train][TEST][Train]  -> Error en test 4
Iteracion 5: [Train][Train][Train][Train][TEST]  -> Error en test 5

Metrica final = Promedio(Error1, Error2, Error3, Error4, Error5)
```

### Ventajas

- **Todos los datos se usan** tanto para entrenar como para evaluar
- **Estimacion mas confiable** del rendimiento real
- **Detecta sobreajuste** si hay mucha diferencia entre train y test

### En este software

El usuario puede elegir entre 2 y 20 folds (por defecto 5). El software usa:

- **StratifiedKFold** para clasificacion: mantiene la proporcion de clases en cada fold
- **KFold** para regresion: division aleatoria normal

---

## Validacion Anidada (Nested Cross-Validation)

### El problema de la validacion simple

Cuando usamos validacion cruzada para **elegir los mejores hiperparametros** (como el numero de arboles o la profundidad), hay un riesgo: el modelo podria "memorizar" los datos de validacion tambien.

### La solucion: dos capas de validacion

```
FASE EXTERNA (evaluacion final):
┌─────────────────────────────────────────┐
│ Fold 1: [TEST]                          │
│ Fold 2: [TRAIN──────────────────]       │
│ Fold 3: [TRAIN──────────────────]       │
│ Fold 4: [TRAIN──────────────────]       │
│ Fold 5: [TRAIN──────────────────]       │
└─────────────────────────────────────────┘

Dentro de cada fold de entrenamiento, se hace
validacion interna para buscar hiperparametros:

FASE INTERNA (optimizacion):
┌─────────────────────────────────────────┐
│ Iter 1: [INT_TEST] [INT_TRAIN]          │
│ Iter 2: [INT_TRAIN] [INT_TEST]          │
│ Iter 3: [INT_TRAIN] [INT_TEST]          │
└─────────────────────────────────────────┘
```

### Flujo completo

1. Se divide en K folds externos
2. Para cada fold externo:
   a. Se toma el entrenamiento de ese fold
   b. Se usa validacion interna para encontrar los mejores hiperparametros
   c. Se entrena con esos hiperparametros en el entrenamiento del fold externo
   d. Se evalua en el test del fold externo
3. Se promedian los resultados de todos los folds externos

### En este software

- **Folds externos:** 5 por defecto (configurable 2-20)
- **Folds internos:** 3 por defecto (configurable 2-10)
- **Modos:** `nested` (validacion anidada, por defecto) o `single` (una sola optimizacion)

---

## Optimizacion de Hiperparametros con Optuna

### Que son los hiperparametros

Son los **ajustes manuales** del modelo que no se aprenden de los datos. Ejemplos:

| Modelo | Hiperparametros |
|--------|-----------------|
| Random Forest | numero de arboles, profundidad maxima, minimo de muestras por hoja |
| XGBoost | tasa de aprendizaje, numero de arboles, profundidad, regularizacion |
| Ridge | alpha (fuerza de regularizacion) |

### El problema: busqueda en el espacio

Si cada hiperparametro tiene 10 posibles valores y hay 5 hiperparametros, hay `10^5 = 100,000` combinaciones. Probar todas es imposible.

### Optuna: busqueda inteligente

Optuna usa el algoritmo **TPE (Tree-structured Parzen Estimator)**, que es una forma inteligente de buscar:

1. Prueba algunas combinaciones al azar
2. Analiza cuales funcionaron mejor
3. **Aprende** que tipos de valores son buenos
4. **Enfoca** la busqueda en las zonas prometedoras

```
Busqueda aleatoria:          Optuna TPE:
. . . . . . . . . .         . . . . . . . . . .
  . .                           . .
    .                             .
      .   .                        **  **  **
        .  .  .                    **  **  **
          .                           **
    Distribuido uniformemente      Concentrado en lo bueno
```

### En este software

| Parametro | Valor por defecto | Rango |
|-----------|-------------------|-------|
| `n_trials` (intentos) | 25 | 1 a 500 |
| `timeout_s` (tiempo maximo) | 3600 segundos (1 hora) | - |
| Algoritmo | TPE Sampler | - |

### Funcion objetivo de Optuna

Para cada combinacion de hiperparametros, Optuna entrena el modelo con validacion interna y mide:

- **Clasificacion:** Maximizar F1 (binary o macro)
- **Regresion:** Minimizar RMSE

La mejor combinacion de hiperparametros es la que maximiza (o minimiza) esta funcion.

---

## Intervalo de Confianza al 95%

### Concepto simple
El software no solo da un numero de metrica, sino un **intervalo** que dice "estamos 95% seguros de que el valor real esta entre A y B".

### Formula matematica

```
IC 95% = media +/- t * (std / sqrt(k))
```

Donde:
- `media` = promedio de la metrica en los K folds
- `std` = desviacion estandar de la metrica en los K folds
- `k` = numero de folds
- `t` = valor critico de la distribucion t de Student con k-1 grados de libertad al nivel 0.975

### Ejemplo

Supongamos que en 5 folds obtuvimos R^2 de: 0.82, 0.85, 0.83, 0.84, 0.86

```
media = 0.84
std = 0.0158 (con ddof=1)
k = 5
t (0.975, df=4) = 2.776

IC = 0.84 +/- 2.776 * (0.0158 / sqrt(5))
   = 0.84 +/- 2.776 * 0.00706
   = 0.84 +/- 0.0196

IC 95% = [0.8204, 0.8596]
```

### Interpretacion

Estamos 95% seguros de que el verdadero R^2 del modelo esta entre 0.8204 y 0.8596. Si el intervalo es muy ancho, significa que el modelo es inestable.

---

## Resumen del proceso de validacion

```
1. Dividir datos en K folds (externos)
         |
2. Para cada fold externo:
   a. Entrenar con K-1 folds
   b. Usar validacion interna para buscar hiperparametros (Optuna)
   c. Re-entrenar con los mejores hiperparametros
   d. Evaluar en el fold de test
         |
3. Promediar metricas de todos los folds
         |
4. Calcular intervalo de confianza al 95%
         |
5. Rankear modelos por metrica principal
         |
6. Entrenar el modelo ganador con TODOS los datos
```
