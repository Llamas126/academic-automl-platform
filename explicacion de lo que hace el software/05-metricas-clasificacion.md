# Metricas de Clasificacion

Para medir que tan bien un modelo clasifica, usamos estas 9 metricas. Primero hay que entender la **matriz de confusion**.

---

## La Matriz de Confusion

Es una tabla que muestra cuantas predicciones fueron correctas e incorrectas para cada clase.

### Para 2 clases (Binaria)

```
                    PREDICCION
                  Positiva  Negativa
REAL  Positiva  |   VP    |   FN    |
      Negativa  |   FP    |   VN    |
```

| Simbolo | Nombre | Que es |
|---------|--------|--------|
| **VP** (True Positive) | Verdadero Positivo | Predijo positivo y era positivo |
| **VN** (True Negative) | Verdadero Negativo | Predijo negativo y era negativo |
| **FP** (False Positive) | Falso Positivo | Predijo positivo pero era negativo (falsa alarma) |
| **FN** (False Negative) | Falso Negativo | Predijo negativo pero era positivo (fallo) |

### Ejemplo: Test de enfermedad

```
                    PREDICCION
                  Enfermo   Sano
REAL  Enfermo  |    85    |   15   |  = 100 enfermos
      Sano     |     5    |  880   |  = 885 sanos

VP = 85  (detecto correctamente los enfermos)
FN = 15  (fallo: dijo sano pero estaba enfermo) <- PELIGROSO
FP = 5   (falso alarmo: dijo enfermo pero estaba sano)
VN = 880 (detecto correctamente los sanos)
```

### Para multiples clases

La matriz se expande pero el principio es el mismo: la diagonal son los aciertos, fuera de la diagonal son errores.

```
              PREDICCION
           A    |    B    |    C
REAL A  |  150  |   20   |   10   |
REAL B  |   15  |  180   |    5   |
REAL C  |    5  |   10   |  170   |
```

---

## 1. Accuracy (Exactitud)

### Concepto simple
Que porcentaje de todas las predicciones fueron correctas.

### Formula

```
Accuracy = (VP + VN) / (VP + VN + FP + FN)
         = predicciones correctas / total de predicciones
```

### Ejemplo

```
Accuracy = (85 + 880) / (85 + 880 + 5 + 15) = 965 / 985 = 0.9797 = 97.97%
```

### Cuando falla

El accuracy es **enganoso** cuando hay desbalance de clases:

```
Ejemplo: 95% sanos, 5% enfermos
Un modelo que SIEMPRE predice "sano" tiene:
Accuracy = 95%  <-- Parece bueno, pero es inutil
```

Por eso se usan las demas metricas.

---

## 2. Balanced Accuracy (Exactitud Balanceada)

### Concepto simple
Promedia el accuracy de cada clase por separado, luego promedia esos valores. Asi, las clases minoritarias tienen el mismo peso que las mayoritarias.

### Formula

```
Balanced Accuracy = (Sensibilidad_Clase1 + Sensibilidad_Clase2 + ... + Sensibilidad_ClaseK) / K
```

Donde la sensibilidad (recall) de cada clase es:

```
Sensibilidad_Clase_k = VP_k / (VP_k + FN_k)
```

### Ejemplo

```
Sensibilidad enfermos = 85 / (85 + 15) = 0.85
Sensibilidad sanos   = 880 / (880 + 5) = 0.9944

Balanced Accuracy = (0.85 + 0.9944) / 2 = 0.9222
```

Comparado con el accuracy normal (0.9797), el balanceado (0.9222) refleja mejor que el modelo falla mas con los enfermos.

---

## 3. Precision

### Concepto simple
De todas las veces que el modelo dijo "positivo", **cuantas veces realmente era positivo**.

### Formula

```
Precision = VP / (VP + FP)
```

### Ejemplo

```
Precision = 85 / (85 + 5) = 85 / 90 = 0.9444
```

**Interpretacion:** Cuando el modelo dice "enfermo", tiene 94.44% de probabilidad de estar en lo correcto.

### Analogia
Es como un detector de humo: la precision es que porcentaje de las veces que suena la alarma, realmente hay incendio. Si suena mucho por cocinar, la precision es baja.

---

## 4. Recall (Sensibilidad)

### Concepto simple
De todos los que **realmente son positivos**, cuantos logro detectar el modelo.

### Formula

```
Recall = VP / (VP + FN)
```

### Ejemplo

```
Recall = 85 / (85 + 15) = 85 / 100 = 0.85
```

**Interpretacion:** El modelo detecto el 85% de los realmente enfermos. Se le escaparon 15 pacientes.

### Precision vs Recall

```
Alta precision, bajo recall:
  El modelo es conservador: solo dice "positivo" cuando esta muy seguro
  -> Pocos falsos positivos, pero se pierde gente

Baja precision, alto recall:
  El modelo es agresivo: dice "positivo" para casi todos
  -> Detecta a todos, pero muchas falsas alarmas
```

---

## 5. F1-Score

### Concepto simple
Es la **media armonica** entre Precision y Recall. Equilibra ambas metricas en un solo numero.

### Formula

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Ejemplo

```
F1 = 2 * (0.9444 * 0.85) / (0.9444 + 0.85)
   = 2 * 0.8027 / 1.7944
   = 0.8947
```

### Por que media armonica y no aritmetica?

```
Precision = 1.0, Recall = 0.1
Media aritmetica: (1.0 + 0.1) / 2 = 0.55  (parece decente)
Media armonica:   2 * (1.0 * 0.1) / (1.0 + 0.1) = 0.182  (refleja que el recall es muy bajo)
```

La media armonica es mas severa: si uno de los dos es bajo, el F1 sera bajo.

### F1 Macro vs Weighted

**En este software** se calculan dos versiones:

- **F1 Macro:** Promedia el F1 de cada clase por igual (sin importar cuantas muestras tiene cada clase)
- **F1 Weighted:** Promedia ponderado por el numero de muestras de cada clase

**La metrica principal para clasificacion en este software es F1 Macro.**

---

## 6. Matthews Correlation Coefficient (MCC)

### Concepto simple
Es una metrica que considera **todas** las celdas de la matriz de confusion (VP, VN, FP, FN). Se considera una de las mejores metricas para clasificacion, especialmente con datos desbalanceados.

### Formula

```
MCC = (VP*VN - FP*FN) / sqrt((VP+FP)(VP+FN)(VN+FP)(VN+FN))
```

### Valores

- **MCC = +1** -> Clasificacion perfecta
- **MCC = 0** -> No mejor que adivinar al azar
- **MCC = -1** -> Clasificacion inversamente perfecta (siempre se equivoca)

### Ejemplo

```
MCC = (85*880 - 5*15) / sqrt((85+5)(85+15)(880+5)(880+15))
    = (74800 - 75) / sqrt(90*100*885*895)
    = 74725 / sqrt(7,153,575,000)
    = 74725 / 84579
    = 0.8834
```

---

## 7. Cohen's Kappa

### Concepto simple
Mide el **acuerdo** entre las predicciones y los valores reales, corregido por el acuerdo que se esperaria por azar.

### Formula

```
Kappa = (Po - Pe) / (1 - Pe)
```

Donde:
- `Po` = acuerdo observado (accuracy)
- `Pe` = acuerdo esperado por azar

### Calculo de Pe (acuerdo por azar)

```
Pe = (prob_real_positivo * prob_pred_positivo) + (prob_real_negativo * prob_pred_negativo)
```

### Ejemplo

```
Po = 0.9797 (accuracy)

prob_real_pos = 100/985 = 0.1015
prob_pred_pos = 90/985 = 0.0914
prob_real_neg = 885/985 = 0.8985
prob_pred_neg = 895/985 = 0.9086

Pe = (0.1015 * 0.0914) + (0.8985 * 0.9086)
   = 0.00928 + 0.8163
   = 0.8256

Kappa = (0.9797 - 0.8256) / (1 - 0.8256)
      = 0.1541 / 0.1744
      = 0.8837
```

### Interpretacion del Kappa

| Valor | Interpretacion |
|-------|----------------|
| < 0 | Peor que azar |
| 0 - 0.20 | Acuerdo muy pobre |
| 0.21 - 0.40 | Acuerdo pobre |
| 0.41 - 0.60 | Acuerdo moderado |
| 0.61 - 0.80 | Buen acuerdo |
| 0.81 - 1.0 | Excelente acuerdo |

---

## 8. ROC AUC (Area Bajo la Curva ROC)

### Concepto simple
La curva ROC grafica que tan bien el modelo separa las clases, considerando **todos los umbrales de decision** posibles.

### Que es la curva ROC

La ROC grafica dos cosas:

```
TPR (Tasa de Verdaderos Positivos) = Recall = VP / (VP + FN)
FPR (Tasa de Falsos Positivos) = FP / (FP + VN)
```

```
TPR
1.0 |            ___________
    |           /
    |          /
    |         /
    |        /
    |       /
    |      /
    |     /
    |    /
0.0 |___/___________________
    0.0                  1.0
              FPR
```

- **Curva pegada a la esquina superior izquierda** = Modelo perfecto
- **Linea diagonal** = Modelo que adivina al azar

### Que es el AUC

El AUC es el **area bajo la curva ROC**. Va de 0 a 1:

- **AUC = 1.0** -> Modelo perfecto
- **AUC = 0.5** -> Modelo aleatorio (no sirve)
- **AUC < 0.5** -> Modelo inversamente peor que azar

### Interpretacion probabilistica

El AUC se puede interpretar como: "Si elijo una muestra positiva y una negativa al azar, el AUC es la probabilidad de que el modelo le dé mayor puntaje a la positiva."

### Para multiples clases

Cuando hay mas de 2 clases, se usa la estrategia **One-vs-Rest**: se entrena un modelo binario para cada clase contra todas las demas, y se promedia el AUC de cada uno (macro-averaging).

---

## 9. Precision/Recall por clase

### Para binario
Se calcula precision y recall para la clase positiva (o se puede especificar).

### Para multiples clases
Se calcula para cada clase y se promedia:

- **Macro:** Promedio simple (todas las clases pesan igual)
- **Weighted:** Promedio ponderado por el numero de muestras de cada clase

**En este software:** Se usa automaticamente `binary` para 2 clases y `macro` para multiples clases.

---

## Resumen de metricas de clasificacion

| Metrica | Que mide | Valores ideales |
|---------|----------|-----------------|
| **Accuracy** | % total de aciertos | Cerca de 1.0 |
| **Balanced Accuracy** | Accuracy promedio por clase | Cerca de 1.0 |
| **Precision** | De los predichos positivos, cuantos lo son | Cerca de 1.0 |
| **Recall** | De los reales positivos, cuantos detecto | Cerca de 1.0 |
| **F1 Score** | Balance entre Precision y Recall | Cerca de 1.0 |
| **MCC** | Correlacion entre prediccion y realidad | Cerca de +1.0 |
| **Kappa** | Acuerdo corregido por azar | Cerca de 1.0 |
| **ROC AUC** | Capacidad de separacion | Cerca de 1.0 |

**En este software:** La metrica principal es **F1 Macro** (mayor es mejor).
