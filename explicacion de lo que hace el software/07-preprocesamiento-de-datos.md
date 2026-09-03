# Preprocesamiento de Datos

Antes de que un modelo pueda aprender, los datos necesitan ser **preparados**. Este proceso se llama preprocesamiento.

---

## Paso 1: Detectar tipos de columna

El software analiza cada columna automaticamente y la clasifica en un rol:

| Rol | Que detecta | Que hace con ella |
|-----|-------------|-------------------|
| **Numerica** | Valores numericos | Imputa valores faltantes, opcionalmente escala |
| **Categorica** | Texto con pocas categorias | Imputa, convierte a numeros (one-hot) |
| **Booleana** | Verdadero/Falso, Si/No | Trata como categorica binaria |
| **Identificador** | IDs unicos (95%+ valores unicos) | **Excluye** del entrenamiento |
| **Fecha/Hora** | Fechas | **Excluye** del entrenamiento |
| **Texto libre** | Texto largo y variable | **Excluye** del entrenamiento |

### Que se excluye

Las columnas de tipo **identificador**, **fecha** y **texto libre** se excluyen porque no son predictoras utiles. Por ejemplo, el "ID" de un paciente no ayuda a predecir su enfermedad.

---

## Paso 2: Manejo de valores faltantes

Los valores faltantes (NaN, vacios) son un problema comun. Los modelos no pueden trabajar con ellos.

### Para variables numericas: Imputacion por mediana

```
Datos originales:  [25, 30, NaN, 45, 28, NaN, 35]
Mediana:           30
Datos imputados:   [25, 30, 30, 45, 28, 30, 35]
```

**Por que mediana y no media?**

```
Datos: [25, 30, NaN, 45, 28, 1000]  (1000 es un outlier)
Media:  225.6  (distorsionada por el outlier)
Mediana: 30   (robusta al outlier)
```

La mediana es mas robusta a valores extremos.

### Para variables categoricas: Imputacion por moda

```
Datos originales:  [rojo, azul, NaN, rojo, verde, NaN]
Moda:              rojo
Datos imputados:   [rojo, azul, rojo, rojo, verde, rojo]
```

Se reemplaza con el valor mas frecuente.

---

## Paso 3: Escalado de variables (opcional)

Algunos modelos necesitan que todas las variables esten en la **misma escala**.

### Cuando es necesario

| Modelo | Necesita escalado? |
|--------|-------------------|
| Regresion Lineal | No (pero ayuda con regularizacion) |
| Ridge / Lasso | **Si** |
| Logistic Regression | **Si** |
| Random Forest | No |
| Gradient Boosting / XGBoost / LightGBM | No |

### StandardScaler (Estandarizacion Z-score)

Convierte cada variable para que tenga **media 0 y desviacion estandar 1**:

```
z = (x - media) / desviacion_estandar
```

**Ejemplo:**

```
Variable "edad": [25, 30, 35, 40, 45]
Media: 35
Desviacion: 7.906

Valores escalados:
25 -> (25-35)/7.906 = -1.265
30 -> (30-35)/7.906 = -0.632
35 -> (35-35)/7.906 =  0.000
40 -> (40-35)/7.906 =  0.632
45 -> (45-35)/7.906 =  1.265
```

**Por que importa para Ridge/Lasso:**

Estas regularizaciones penalizan la magnitud de los coeficientes. Si una variable tiene valores entre 0 y 1, y otra entre 0 y 1,000,000, la segunda tendra un coeficiente mucho mas grande naturalmente, no porque sea mas importante sino por la escala.

---

## Paso 4: Codificacion One-Hot (One-Hot Encoding)

Los modelos solo entienden numeros, no texto. El one-hot encoding convierte categorias en columnas de 0 y 1.

### Ejemplo

**Variable original "Color":** [rojo, azul, verde, rojo, azul]

**Despues de one-hot:**

| Color_rojo | Color_azul | Color_verde |
|-----------|-----------|------------|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |

### Manejo de categorias raras

El software agrupa categorias que aparecen en menos del 1% de los datos en una categoria "unknown":

```
Datos: [gato, perro, gato, pez, gato, hamster, gato, ...]
                                                    (1000 filas)

gato:     850 veces (85%)  -> columna gato
perro:    100 veces (10%)  -> columna perro
pez:       30 veces (3%)   -> columna pez
hamster:   20 veces (2%)   -> columna hamster
raro1:      1 veces (0.1%) -> unknown
raro2:      1 veces (0.1%) -> unknown
```

Esto evita crear columnas para categorias con muy pocos ejemplos.

---

## Pipeline completo de preprocesamiento

El software construye un `ColumnTransformer` que aplica transformaciones diferentes segun el tipo de columna:

```
Entrada: [edad, ingresos, color, tamaño]
              |        |       |       |
              v        v       v       v
         Numerico  Numerico  Cat     Cat
              |        |       |       |
              v        v       v       v
         Imputar   Imputar  Imputar  Imputar
         (mediana) (mediana) (moda)   (moda)
              |        |       |       |
              v        v       v       v
         [Si aplica] [Si aplica] One-Hot  One-Hot
         Escalar     Escalar    Encoding Encoding
              |        |       |       |
              v        v       v       v
         [edad_scaled, ingresos_scaled, color_rojo, color_azul, tamaño_S, tamaño_M, tamaño_L]
```

### Pipeline completo del modelo

El preprocesador se combina con el modelo en un solo `Pipeline`:

```
Pipeline:
  Paso 1: preprocesador (transformar datos)
  Paso 2: modelo (aprender patron)
```

Esto garantiza que durante la prediccion, los nuevos datos se transforman **exactamente igual** que los datos de entrenamiento.

---

## Eliminacion de filas con target faltante

Si la variable objetivo (lo que queremos predecir) tiene valores faltantes, esas filas se eliminan completamente. No tiene sentido entrenar con datos donde no sabemos cual es la respuesta correcta.

```
Datos originales: 1000 filas
Filas con target NaN: 15
Filas para entrenar: 985
```
