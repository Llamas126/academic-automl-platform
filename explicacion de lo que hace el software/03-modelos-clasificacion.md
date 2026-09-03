# Modelos de Clasificacion

Este software implementa **5 modelos de clasificacion**. Mientras que la regresion predice numeros, la clasificacion predice **categorias**.

---

## 1. Regresion Logistica

### Concepto simple
A pesar de llamarse "regresion", es un modelo de **clasificacion**. No predice un numero directamente, sino la **probabilidad** de que algo pertenezca a una clase.

### La funcion sigmoid

La regresion logistica usa la **funcion sigmoide** para convertir cualquier numero en una probabilidad entre 0 y 1:

```
sigma(z) = 1 / (1 + e^(-z))
```

Donde `z` es la misma combinacion lineal que en la regresion lineal:

```
z = b0 + b1*x1 + b2*x2 + ... + bp*xp
```

### Visualizacion de la sigmoid

```
Probabilidad
    1.0 |                    ............
        |                 ...
        |               ..
    0.5 |             .        <-- frontera de decision
        |           ..
        |         ...
    0.0 |............
        +-------------------------> z
       -6   -4   -2    0    2    4    6
```

### Regla de decision

```
Si sigma(z) >= 0.5  ->  Clase 1 (ej: "aprobado")
Si sigma(z) <  0.5  ->  Clase 0 (ej: "reprobado")
```

### Para multiples clases (Softmax)

Cuando hay mas de 2 clases, se usa la **funcion softmax**:

```
P(clase k) = e^(zk) / Suma(j=1 a K) de e^(zj)
```

Donde `K` es el numero total de clases. Softmax garantiza que todas las probabilidades sumen 1.

### Hiperparametros

| Hiperparametro | Que controla | Rango buscado |
|---------------|--------------|---------------|
| `C` | Inversa de la regularizacion (mientras mas grande, menos regularizacion) | 0.001 a 1000 (log) |

**En este software:** Se usa `max_iter=5000` para asegurar convergencia.

---

## 2. Random Forest Classifier

### Concepto simple
Igual que el Random Forest de regresion, pero en lugar de promediar predicciones numericas, cada arbol **vota por una clase** y gana la clase con mas votos.

### Formula

```
Prediccion = moda(T1(x), T2(x), ..., TB(x))
```

Donde `moda()` es la clase que aparece mas veces entre los B arboles.

### Ejemplo con 5 arboles

```
Arbol 1 predice: Aprobado
Arbol 2 predice: Aprobado
Arbol 3 predice: Reprobado
Arbol 4 predice: Aprobado
Arbol 5 predice: Aprobado

Resultado: Aprobado (4 votos contra 1)
```

### Hiperparametros

| Hiperparametro | Rango |
|---------------|-------|
| `n_estimators` | 100 a 500 |
| `max_depth` | 4, 6, 8, 12, 16, 24, o sin limite |
| `min_samples_split` | 2 a 20 |
| `min_samples_leaf` | 1 a 8 |
| `max_features` | sqrt, log2, o 50% |

### Nota especial
Se usa `class_weight="balanced"`, lo que le da **mas peso** a las clases que tienen menos ejemplos. Esto es util cuando hay desbalance (ej: 900 aprobados y 100 reprobados).

---

## 3. Gradient Boosting Classifier

### Concepto simple
Igual que el Gradient Boosting de regresion, pero adaptado para clasificacion. Cada arbol nuevo corrige los errores de clasificacion del anterior.

### Como funciona con probabilidades

1. Se calcula la probabilidad base de cada clase
2. Los residuos son: real - probabilidad_predicha
3. Cada arbol nuevo ajusta esas probabilidades
4. La prediccion final es la clase con mayor probabilidad acumulada

### Hiperparametros

| Hiperparametro | Rango |
|---------------|-------|
| `learning_rate` | 0.01 a 0.3 (log) |
| `n_estimators` | 50 a 400 |
| `max_depth` | 2 a 6 |
| `subsample` | 0.5 a 1.0 |

---

## 4. XGBoost Classifier

### Concepto simple
Version optimizada de Gradient Boosting para clasificacion. Usa la funcion de perdida **logloss** (perdida logaritmica) para medir que tan bien predice probabilidades.

### Funcion de perdida logloss

```
LogLoss = -(1/n) * Suma(i=1 a n) de [yi * log(pi) + (1-yi) * log(1-pi)]
```

Donde:
- `yi` = 1 si es clase positiva, 0 si es negativa
- `pi` = probabilidad predicha de ser clase positiva
- `log` = logaritmo natural

**Interpretacion:** A menor LogLoss, mejor es el modelo. Un modelo perfecto tendria LogLoss = 0.

### Hiperparametros

| Hiperparametro | Rango |
|---------------|-------|
| `n_estimators` | 100 a 800 |
| `learning_rate` | 0.01 a 0.3 (log) |
| `max_depth` | 3 a 10 |
| `subsample` | 0.5 a 1.0 |
| `colsample_bytree` | 0.5 a 1.0 |
| `reg_lambda` | 0.01 a 10 (log) |
| `reg_alpha` | 0.001 a 10 (log) |

**Nota:** Se usa `eval_metric="logloss"` que es la metrica que XGBoost usa internamente para decidir como dividir los arboles.

---

## 5. LightGBM Classifier

### Concepto simple
LightGBM para clasificacion. Usa la tecnica **leaf-wise** (crecimiento por hojas) que lo hace mas rapido que los demas.

### Hiperparametros

| Hiperparametro | Rango |
|---------------|-------|
| `n_estimators` | 100 a 800 |
| `learning_rate` | 0.01 a 0.3 (log) |
| `num_leaves` | 8 a 128 (log) |
| `min_child_samples` | 5 a 50 |
| `colsample_bytree` | 0.5 a 1.0 |
| `reg_lambda` | 0.01 a 10 (log) |

**Nota:** Se usa `class_weight="balanced"` como en Random Forest para manejar el desbalance de clases.

---

## Comparacion de modelos de clasificacion

```
Modelo               | Velocidad | Precision tipica | Interpretabilidad
---------------------|-----------|------------------|------------------
Regresion Logistica  | Muy rapido| Moderada         | Alta (coeficientes)
Random Forest        | Moderado  | Buena            | Media (feature importance)
Gradient Boosting    | Lento     | Muy buena        | Baja
XGBoost              | Rapido*   | Muy buena        | Baja
LightGBM             | Muy rapido| Muy buena        | Baja

* Rapido despues de ser optimizado
```

## Arboles de Decision: El bloque fundamental

Todos los modelos basados en arboles (Random Forest, GBM, XGBoost, LightGBM) comparten el mismo bloque fundamental: el **arbol de decision**.

### Como funciona un arbol para clasificar

```
                     Presion > 140?
                    /              \
                 SI                 NO
                /                    \
        Edad > 50?              Riesgo BAJO
        /         \
      SI           NO
      /              \
Riesgo ALTO    Presion > 130?
              /           \
            SI             NO
            /                \
    Riesgo MEDIO        Riesgo BAJO
```

En cada nodo, el arbol elige la pregunta que **mejor separa** las clases. Para medir la "calidad" de una separacion, se usa la **entropia** o el **indice Gini**.

### Entropia

```
Entropia(S) = -Suma(k=1 a K) de pk * log2(pk)
```

Donde `pk` es la fraccion de muestras de la clase k en el conjunto S.

- Entropia = 0: Todas las muestras son de la misma clase (puro)
- Entropia = maxima: Hay igual proporcion de todas las clases (impuro)

### Indice Gini

```
Gini(S) = 1 - Suma(k=1 a K) de pk^2
```

- Gini = 0: Conjunto perfectamente puro
- Gini = maxima: Maximum impureza
