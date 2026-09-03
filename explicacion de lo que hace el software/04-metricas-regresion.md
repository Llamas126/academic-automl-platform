# Metricas de Regresion

Cuando un modelo predice un numero, necesitamos medir **que tan cerca** esta la prediccion del valor real. Estas son las 6 metricas que el software calcula.

---

## 1. R-cuadrado (R^2) - Coeficiente de Determinacion

### Concepto simple
R^2 mide **que porcentaje de la variacion del target** es explicado por el modelo. Va de menos infinito a 1.

- **R^2 = 1.0** -> El modelo explica el 100% de la variacion (prediccion perfecta)
- **R^2 = 0.0** -> El modelo es tan bueno como simplemente predecir el promedio
- **R^2 < 0** -> El modelo es peor que predecir el promedio

### Formula matematica

```
R^2 = 1 - (SS_res / SS_tot)
```

Donde:

**SS_res** (suma de cuadrados de los residuos):
```
SS_res = Suma(i=1 a n) de (yi - yi_pred)^2
```
Es la suma de los errores al cuadrado. Mide cuanto "falla" el modelo.

**SS_tot** (suma de cuadrados total):
```
SS_tot = Suma(i=1 a n) de (yi - y_barra)^2
```
Donde `y_barra` es el promedio del target. Mide la variacion natural de los datos.

### Ejemplo numerico

Supongamos que queremos predecir precios de casas:

| Casa | Real (yi) | Predicho (yi_pred) | Error (yi - yi_pred) | Error^2 |
|------|-----------|--------------------|--------------------|---------|
| 1 | $100,000 | $105,000 | -$5,000 | 25,000,000 |
| 2 | $150,000 | $142,000 | +$8,000 | 64,000,000 |
| 3 | $200,000 | $198,000 | +$2,000 | 4,000,000 |
| 4 | $250,000 | $260,000 | -$10,000 | 100,000,000 |

Promedio real: y_barra = $175,000

```
SS_res = 25M + 64M + 4M + 100M = 193,000,000

SS_tot = (100k-175k)^2 + (150k-175k)^2 + (200k-175k)^2 + (250k-175k)^2
       = 5,625M + 625M + 625M + 5,625M
       = 12,500,000,000

R^2 = 1 - (193M / 12,500M) = 1 - 0.01544 = 0.9846
```

**Interpretacion:** El modelo explica el 98.46% de la variacion en los precios.

---

## 2. R^2 Ajustado

### Concepto simple
El R^2 normal **siempre mejora** si agregamos mas variables, aunque esas variables no sirvan. El R^2 ajustado corrige eso penalizando las variables que no aportan.

### Formula matematica

```
R^2_ajustado = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
```

Donde:
- `n` = numero de muestras (filas de datos)
- `p` = numero de features (columnas de entrada)
- `R^2` = el R^2 normal

### Ejemplo

Si `R^2 = 0.85`, `n = 100` muestras, `p = 10` features:

```
R^2_ajustado = 1 - (1 - 0.85) * (100 - 1) / (100 - 10 - 1)
             = 1 - 0.15 * 99 / 89
             = 1 - 0.1669
             = 0.8331
```

El R^2 ajustado es menor que el R^2 normal, penalizando las 10 features.

### Cuando es importante

Si tenes 20 features y R^2 = 0.90 pero R^2_ajustado = 0.70, significa que varias de esas features no aportan nada al modelo.

---

## 3. RMSE (Error Cuadratico Medio)

### Concepto simple
Es la **raiz del promedio de los errores al cuadrado**. Tiene la misma unidad que el target, lo que facilita la interpretacion.

### Formula matematica

```
RMSE = sqrt( (1/n) * Suma(i=1 a n) de (yi - yi_pred)^2 )
```

### Ejemplo

Usando el ejemplo anterior:

```
RMSE = sqrt(193,000,000 / 4) = sqrt(48,250,000) = $6,946
```

**Interpretacion:** En promedio, el modelo se equivoca por aproximadamente $6,946 en la prediccion del precio.

### Por que se usa sqrt?

Porque la suma de errores al cuadrado (SSE) esta en unidades al cuadrado (dolares^2), que no tiene sentido. Al sacar la raiz, volvemos a dolares.

### RMSE vs MSE

| MSE (Error Cuadratico Medio) | RMSE |
|------------------------------|------|
| `(1/n) * Suma(yi - yi_pred)^2` | `sqrt(MSE)` |
| Unidades: dolares^2 | Unidades: dolares |
| Dificil de interpretar | Facil de interpretar |

**En este software:** RMSE es la metrica principal para regresion. Los modelos se rankean por RMSE (menor es mejor).

---

## 4. MAE (Error Absoluto Medio)

### Concepto simple
Es el promedio de los **errores absolutos**. A diferencia del RMSE, no penaliza mas los errores grandes.

### Formula matematica

```
MAE = (1/n) * Suma(i=1 a n) de |yi - yi_pred|
```

### Ejemplo

| Casa | Real | Predicho | Error absoluto |
|------|------|----------|---------------|
| 1 | $100k | $105k | $5,000 |
| 2 | $150k | $142k | $8,000 |
| 3 | $200k | $198k | $2,000 |
| 4 | $250k | $260k | $10,000 |

```
MAE = (5000 + 8000 + 2000 + 10000) / 4 = $6,250
```

### RMSE vs MAE

| RMSE | MAE |
|------|-----|
| Penaliza mas los errores grandes (por el cuadrado) | Trata todos los errores por igual |
| Mas sensible a outliers | Mas robusto a outliers |
| Si RMSE >> MAE, hay errores grandes puntuales | Si RMSE ~= MAE, los errores son uniformes |

---

## 5. MAPE (Error Porcentual Absoluto Medio)

### Concepto simple
Es el promedio de los errores en **porcentaje**. Es muy intuitivo porque dice "en promedio, el modelo se equivoca por X%".

### Formula matematica

```
MAPE = (1/n) * Suma(i=1 a n) de |yi - yi_pred| / |yi| * 100
```

**Nota:** Se excluyen las muestras donde `yi = 0` porque no se puede dividir por cero.

### Ejemplo

| Casa | Real | Predicho | Error % |
|------|------|----------|---------|
| 1 | $100k | $105k | 5% |
| 2 | $150k | $142k | 5.33% |
| 3 | $200k | $198k | 1% |
| 4 | $250k | $260k | 4% |

```
MAPE = (5 + 5.33 + 1 + 4) / 4 = 3.83%
```

**Interpretacion:** El modelo se equivoca en promedio un 3.83% del valor real.

### Limitaciones

- No funciona si el target tiene valores cercanos a cero
- Penaliza mas los errores por encima del valor real que por debajo

---

## 6. Max Error (Error Maximo)

### Concepto simple
Es el **error mas grande** que comete el modelo en toda la muestra. Es util para saber el peor escenario.

### Formula

```
MaxError = max(|yi - yi_pred|) para todas las muestras i
```

### Ejemplo

Del ejemplo anterior, el error maximo es $10,000 (casa 4).

**Interpretacion:** En el peor caso, el modelo se equivoca por $10,000.

### Para que sirve

Si bien el RMSE o MAE pueden ser bajos, el Max Error puede ser alto. Esto indica que hay al menos una prediccion muy mala. Es importante en contextos donde el error maximo es critico (ej: predecir la presion de un tanque industrial).

---

## Resumen de metricas de regresion

| Metrica | Formula resumida | Que mide | Direction |
|---------|-----------------|----------|-----------|
| **R^2** | `1 - SS_res/SS_tot` | % de variacion explicada | mayor es mejor |
| **R^2 ajustado** | `1 - (1-R^2)(n-1)/(n-p-1)` | R^2 corregido por features | mayor es mejor |
| **RMSE** | `sqrt(mean(error^2))` | Error promedio (penaliza errores grandes) | menor es mejor |
| **MAE** | `mean(abs(error))` | Error promedio (uniforme) | menor es mejor |
| **MAPE** | `mean(abs(error/real)*100)` | Error en porcentaje | menor es mejor |
| **Max Error** | `max(abs(error))` | Peor error cometido | menor es mejor |

**En este software:** La metrica principal para seleccionar el mejor modelo es **RMSE** (menor es mejor).
