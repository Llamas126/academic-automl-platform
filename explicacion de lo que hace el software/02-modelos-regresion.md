# Modelos de Regresion

Este software implementa **7 modelos de regresion**. Cada uno tiene una forma diferente de aprender la relacion entre las caracteristicas y el target.

---

## 1. Regresion Lineal

### Concepto simple
Es el modelo mas basico. Dibuja la **mejor linea recta** que pasa lo mas cerca posible de todos los puntos de datos.

### Formula matematica

```
y = b0 + b1*x1 + b2*x2 + ... + bp*xp
```

Donde:
- `y` = la prediccion (ej: precio de la casa)
- `b0` = intercepto (donde la linea corta el eje Y)
- `b1, b2, ... bp` = coeficientes (pesos de cada variable)
- `x1, x2, ... xp` = las caracteristicas de entrada

### Como se calculan los coeficientes

Se minimiza la **suma de errores cuadrados** (OLS - Ordinary Least Squares):

```
Minimizar: Suma(i=1 a n) de (yi - (b0 + b1*xi1 + ... + bp*xip))^2
```

O en notacion matricial:

```
Minimizar: ||Y - Xb||^2
```

La solucion exacta es:

```
b = (X^T * X)^(-1) * X^T * Y
```

Donde `X^T` es la transpuesta de la matriz de caracteristicas.

### Ventajas y desventajas

| Ventaja | Desventaja |
|---------|------------|
| Facil de entender | Solo captura relaciones lineales |
| Muy rapido | Sensible a valores atipicos (outliers) |
| Los coeficientes se pueden interpretar | Si las variables estan correlacionadas, falla |

---

## 2. Regresion Ridge (L2)

### Concepto simple
Es igual a la regresion lineal pero con una **penalizacion** para que los coeficientes no sean demasiado grandes. Esto evita que el modelo se "memorice" los datos de entrenamiento.

### Formula matematica

```
Minimizar: ||Y - Xb||^2 + alpha * Suma(j=1 a p) de bj^2
```

La diferencia con la lineal es el termino `alpha * Suma(bj^2)`, que penaliza los coeficientes grandes.

### Que hace alpha

- `alpha = 0` -> Es exactamente la regresion lineal normal
- `alpha muy grande` -> Los coeficientes se vuelven muy pequenos (casi cero), el modelo es muy simple
- `alpha optimal` -> Equilibrio entre ajuste y simplicidad

**En este software:** Optuna busca el mejor `alpha` entre `0.001` y `100` en escala logaritmica.

### Analogia
Es como un profesor que le dice al modelo: "si vas a usar un coeficiente muy grande, ten una buena razon para ello".

---

## 3. Regresion Lasso (L1)

### Concepto simple
Similar a Ridge, pero la penalizacion puede **eliminar variables** poniendo sus coeficientes en exactamente cero.

### Formula matematica

```
Minimizar: ||Y - Xb||^2 + alpha * Suma(j=1 a p) de |bj|
```

La diferencia con Ridge es que usa el **valor absoluto** `|bj|` en lugar del cuadrado `bj^2`.

### Ridge vs Lasso

| Ridge (L2) | Lasso (L1) |
|------------|------------|
| Reduce los coeficientes pero nunca los hace cero | Puede hacer coeficientes = 0 (elimina variables) |
| Penaliza el cuadrado de los pesos | Penaliza el valor absoluto de los pesos |
| Bueno cuando todas las variables son utiles | Bueno cuando solo algunas variables importan |

**En este software:** Optuna busca el mejor `alpha` entre `0.0001` y `10` en escala logaritmica. Se usa `max_iter=50000` para asegurar convergencia.

---

## 4. Random Forest Regressor

### Concepto simple
En lugar de dibujar una sola linea, construye **muchos arboles de decision** y promedia sus predicciones. Cada arbol "vota" y se toma el promedio.

### Como funciona un arbol de decision

Un arbol hace preguntas sucesivas sobre las caracteristicas:

```
                  metro^2 > 100?
                 /              \
              SI                 NO
              /                   \
    habitaciones > 3?       precio = $120,000
       /          \
     SI            NO
      /              \
precio=$250,000  precio=$180,000
```

### Formula matematica

```
Prediccion = (1/B) * Suma(b=1 a B) de Tb(x)
```

Donde:
- `B` = numero de arboles (por defecto 100 a 500 en el software)
- `Tb(x)` = prediccion del arbol b para la entrada x

### Hiperparametros (los ajustes que hace el software)

| Hiperparametro | Que controla | Rango buscado |
|---------------|--------------|---------------|
| `n_estimators` | Cuantos arboles construir | 100 a 500 |
| `max_depth` | Que tan profundo es cada arbol | 4, 6, 8, 12, 16, 24, o sin limite |
| `min_samples_split` | Minimo de muestras para dividir un nodo | 2 a 20 |
| `min_samples_leaf` | Minimo de muestras en una hoja | 1 a 8 |
| `max_features` | Cuantas variables considera por arbol | sqrt, log2, o 50% |

### Analogia
Es como preguntarle a 500 personas con experiencia y promediar sus respuestas. El promedio de muchas opiniones suele ser mejor que la opinion de una sola persona.

---

## 5. Gradient Boosting Regressor

### Concepto simple
Construye arboles **uno despues de otro**, donde cada arbol nuevo intenta **corregir los errores** del anterior.

### Formula matematica

El modelo final es una suma de arboles:

```
F(x) = Suma(m=1 a M) de v * hm(x)
```

Donde:
- `M` = numero de arboles (50 a 400 en el software)
- `hm(x)` = el arbol m-esimo
- `v` = tasa de aprendizaje (learning rate, entre 0.01 y 0.3)

### Como se construye

1. Se empieza con una prediccion base (el promedio del target)
2. Se calculan los **residuos** (errores = real - prediccion)
3. Se entrena un arbol para predecir esos residuos
4. Se suma el arbol (multiplicado por el learning rate) al modelo
5. Se repite desde el paso 2

```
Iteracion 1: F1(x) = media(Y) + v * h1(x)
Iteracion 2: F2(x) = F1(x) + v * h2(x)
Iteracion 3: F3(x) = F2(x) + v * h3(x)
...
```

### El learning rate (tasa de aprendizaje)

- `v = 0.3` -> Cada arbol tiene mucho impacto (rapido pero puede sobreajustar)
- `v = 0.01` -> Cada arbol tiene poco impacto (lento pero mas preciso)

**En este software:** Optuna busca el learning rate optimo entre 0.01 y 0.3 en escala logaritmica.

---

## 6. XGBoost Regressor

### Concepto simple
XGBoost es una version **optimizada y mejorada** de Gradient Boosting. Es uno de los algoritmos mas usados en competiciones de datos y en la industria.

### Que lo hace diferente

| Caracteristica | Gradient Boosting | XGBoost |
|---------------|-------------------|---------|
| Velocidad | Normal | Muy rapido (usa GPU y paralelismo) |
| Regularizacion | Ninguna o basica | L1 + L2 integradas |
| Valores faltantes | Requiere imputarlos | Los maneja solo |
| Proteccion contra sobreajuste | Basica | Avanzada |

### Formula matematica (similar pero con regularizacion)

```
Objetivo = Suma(i=1 a n) L(yi, yi_pred) + Suma(k=1 a K) Omega(fk)
```

Donde:
- `L(yi, yi_pred)` = funcion de perdida (que tan mal predice)
- `Omega(fk)` = complejidad del arbol k (penaliza arboles complejos)
- `K` = numero total de arboles

### Hiperparametros buscados

| Hiperparametro | Rango |
|---------------|-------|
| `n_estimators` | 100 a 800 |
| `learning_rate` | 0.01 a 0.3 (log) |
| `max_depth` | 3 a 10 |
| `subsample` | 0.5 a 1.0 (que fraccion de datos usa por arbol) |
| `colsample_bytree` | 0.5 a 1.0 (que fraccion de features usa por arbol) |
| `reg_lambda` | 0.01 a 10 (regularizacion L2) |
| `reg_alpha` | 0.001 a 10 (regularizacion L1) |

---

## 7. LightGBM Regressor

### Concepto simple
LightGBM es otro algoritmo de boosting rapido, pero usa una tecnica diferente para construir los arboles: crece las hojas (**leaf-wise**) en lugar de por niveles (**level-wise**).

### Diferencia visual

```
Level-wise (XGBoost):          Leaf-wise (LightGBM):
        [Nodo]                       [Nodo]
       /      \                     /      \
    [N1]      [N2]              [N1]      [N2]
    / \       / \               / \         \
  [L1][L2] [L3][L4]         [L1][L2]      [L3]
```

Leaf-wise crece solo la rama que mas reduce el error, lo que lo hace mas eficiente en memoria y velocidad.

### Hiperparametros

| Hiperparametro | Rango | Que controla |
|---------------|-------|--------------|
| `n_estimators` | 100 a 800 | Numero de arboles |
| `learning_rate` | 0.01 a 0.3 (log) | Impacto de cada arbol |
| `num_leaves` | 8 a 128 (log) | Complejidad del arbol (hojas) |
| `min_child_samples` | 5 a 50 | Minimo de datos en una hoja |
| `colsample_bytree` | 0.5 a 1.0 | Features por arbol |
| `reg_lambda` | 0.01 a 10 (log) | Regularizacion L2 |

---

## Resumen visual: Como se comparan los modelos

```
Complejidad creciente -->

Lineal    Ridge    Lasso    RandomForest    GBM    XGBoost    LightGBM
  |         |        |           |            |        |           |
 Simple   Simple   Simple    Moderado     Complejo  Complejo   Complejo
 Rapido   Rapido   Rapido    Moderado     Lento*    Rapido*    Rapido

 * GBM es lento, pero XGBoost y LightGBM estan optimizados
```
