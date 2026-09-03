# SHAP y Explicabilidad de Modelos

## El problema: cajas negras

Los modelos mas potentes (Random Forest, XGBoost, LightGBM) son como **cajas negras**: dan buenas predicciones, pero no explican **por que**. Esto es un problema en ciencia, medicina y areas donde se necesita justificar cada decision.

### Ejemplo

Un modelo predice que un paciente tiene 85% de probabilidad de enfermedad cardiaca. El doctor pregunta: "¿Por que?". El modelo no responde.

**SHAP** responde esa pregunta.

---

## Que es SHAP?

**SHAP** (SHapley Additive exPlanations) es un metodo que **explica cada prediccion individual** asignando un valor a cada caracteristica, indicando cuanto contribuyo a la prediccion final.

### Concepto de valor de Shapley

El concepto viene de la **teoria de juegos** (matematicas puras). Imagina que un grupo de personas juegan juntas y obtienen una recompensa. ¿Como se divide la recompensa "justamente"?

Los **valores de Shapley** (1953, Lloyd Shapley, Premio Nobel de Economia) calculan la contribucion promedio de cada jugador considerando **todas las posibles combinaciones** de jugadores.

### SHAP en machine learning

En ML, cada "jugador" es una **caracteristica** (feature), y la "recompensa" es la prediccion.

```
Prediccion base (promedio): 50,000

Contribuciones:
  edad:        +15,000  (aumenta el precio)
  metro^2:     +80,000  (aumenta mucho el precio)
  zona_sur:    -5,000   (baja un poco el precio)
  antiguedad:  -10,000  (baja el precio)

Prediccion final: 50,000 + 15,000 + 80,000 - 5,000 - 10,000 = $130,000
```

### Formula matematica

Para una prediccion con n características:

```
phi_j = Suma(S ⊆ N \ {j}) de [|S|! * (n - |S| - 1)! / n!] * [f(S ∪ {j}) - f(S)]
```

Donde:
- `phi_j` = valor SHAP de la caracteristica j
- `N` = conjunto de todas las caracteristicas
- `S` = subconjunto de caracteristicas (sin j)
- `f(S)` = prediccion del modelo usando solo las caracteristicas de S
- `|S|` = numero de caracteristicas en S
- `n!` = factorial de n

**En la practica**, SHAP no calcula esto literalmente (seria exponencial), sino que usa algoritmos eficientes segun el tipo de modelo.

---

## Estrategias de SHAP que usa el software

El software elige automaticamente el explicador segun el tipo de modelo:

### 1. TreeExplainer (para arboles)

**Modelos:** Random Forest, Gradient Boosting, XGBoost, LightGBM

Es el mas rapido y preciso. Explota la estructura de arbol para calcular valores SHAP exactos en tiempo polinomial.

### 2. LinearExplainer (para modelos lineales)

**Modelos:** Regresion Lineal, Ridge, Lasso, Logistic Regression

Asume relaciones lineales y calcula SHAP de forma directa.

### 3. KernelExplainer (fallback)

**Para cualquier otro modelo.** Es el mas general pero el mas lento. Usa un metodo de aproximacion con un subconjunto de datos de fondo (background samples).

---

## Que genera el software

El software produce 3 tipos de graficos SHAP:

### 1. Bar Plot (Importancia Global)

Muestra **que caracteristicas son mas importantes** en promedio para todo el dataset.

```
Feature Importance (mean |SHAP value|)

metro^2      ████████████████████████  0.45
habitaciones ██████████████            0.28
zona_sur     ████████                  0.16
antiguedad   ████                      0.08
```

**Formula:** `importancia_j = promedio(|valor_SHAP_j|) sobre todas las muestras`

### 2. Beeswarm Plot (Resumen Global + Direccion)

Muestra como **cada valor** de cada caracteristica afecta la prediccion.

```
SHAP beeswarm:

metro^2  ● ● ● ● ● ● ● ●   (valores altos = SHAP positivo)
         ● ● ● ● ● ● ● ●
         ● ● ● ● ● ● ● ●     ● = una muestra
         ● ● ● ● ●
         ● ● ● ●              Puntos a la derecha = aumentan la prediccion
         ● ● ●                Puntos a la izquierda = disminuyen la prediccion

zona     ● ● ● ● ● ● ●
         ● ● ● ● ● ●           Color ROJO = valor alto de la feature
         ● ● ● ●               Color AZUL = valor bajo de la feature
```

### 3. Dependence Plot (Relacion de una feature)

Muestra la **relacion exacta** entre una caracteristica y su efecto en la prediccion.

```
SHAP value para "metro^2"

  +40000 |                    *  *
         |                  *  *  *
  +20000 |              *  *  *
         |           *  *  *
       0 |     *  *  *
         |  *  *
  -20000 |*
         +----------------------------> metro^2
          40   60   80  100  120  140
```

Interpretacion: A partir de cierto tamano, cada metro cuadrado adicional aumenta significativamente el precio.

---

## Propiedades matematicas de SHAP

Los valores SHAP unicos satisfacen estas propiedades:

1. **Eficiencia (Accuracy):** Las contribuciones suman la prediccion menos la prediccion base:
   ```
   Suma(phi_j) = f(x) - E[f(X)]
   ```

2. **Simetria:** Si dos caracteristicas contribuyen igual, reciben el mismo valor SHAP.

3. **Aditividad:** Los valores SHAP se suman para dar la prediccion final.

Estas propiedades son **unicas** de SHAP (son los axiomas de Shapley de 1953).

---

## Explicabilidad vs Precision

| Enfoque | Modelo | Explicable? |
|---------|--------|-------------|
| Caja negra | XGBoost | Alto poder predictivo, baja explicabilidad |
| Lineal | Regresion Lineal | Bajo poder predictivo, alta explicabilidad |
| **SHAP + Caja negra** | **XGBoost + SHAP** | **Alto poder predictivo + explicabilidad post-hoc** |

SHAP permite tener lo mejor de ambos mundos: usar el modelo mas potente y explicar sus decisiones despues.

---

## SHAP en el contexto de este software

1. Despues de entrenar el modelo ganador, el software ejecuta SHAP
2. Genera graficos de importancia global, beeswarm y dependencia
3. Los graficos se incluyen en el informe PDF y en la galeria de la interfaz web
4. El usuario puede ver **por que** el modelo tomo cada decision

Esto es especialmente util para publicaciones academicas, donde la explicabilidad del modelo es un requisito.
