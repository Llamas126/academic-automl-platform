# Glosario de Terminos

Glossario alfabetico de todos los terminos tecnicos usados en el software, explicados para alguien sin formacion en programacion o ciencia de datos.

---

## A

**Accuracy (Exactitud)**
Porcentaje de predicciones correctas sobre el total. Ejemplo: si el modelo predijo 100 cosas y 95 estaban bien, el accuracy es 95%. No es buena metrica cuando hay desbalance de clases.

**Algoritmo**
Receta o conjunto de instrucciones paso a paso para resolver un problema. En ML, es la "formula" que el modelo usa para aprender de los datos.

**Aprendizaje Automatico (Machine Learning)**
Rama de la inteligencia artificial donde las computadoras aprenden patrones a partir de datos, sin programar reglas explicitamente.

**Arbol de Decision**
Modelo que hace preguntas sucesivas para clasificar o predecir. Como un diagrama de flujo: en cada nodo se hace una pregunta y se sigue la rama correspondiente.

**AutoML**
Abreviatura de "Automated Machine Learning". Proceso de automatizar la seleccion de modelos, hiperparametros y preprocesamiento.

---

## B

**Backpropagation (Retropropagacion)**
Algoritmo para entrenar redes neuronales (no usado en este software, pero comun en deep learning).

**Baseline (Linea base)**
Modelo simple usado como punto de comparacion. Ejemplo: predecir siempre el promedio.

**Beeswarm Plot**
Grafico que muestra como cada caracteristica afecta las predicciones de todas las muestras. Cada punto es una prediccion individual.

**Bias (Sesgo)**
Error del modelo por ser demasiado simple. Un modelo con mucho bias no puede capturar patrones complejos (underfitting).

---

## C

**Categorical (Categorica)**
Variable que representa categorias o grupos (ej: "rojo", "azul", "verde"). No tiene significado numerico.

**Class Weight (Peso de clase)**
Ajuste que le da mas importancia a las clases con menos ejemplos. Evita que el modelo ignore las clases minoritarias.

**Clasificacion**
Tipo de problema de ML donde se predice una categoria (ej: spam/no spam, aprobado/reprobado).

**Clustering (Agrupacion)**
Tipo de ML no supervisado donde se agrupan datos similares (no usado en este software).

**CNN (Convolutional Neural Network)**
Red neuronal para procesar imagenes (no usado en este software).

**Cross-Validation (Validacion Cruzada)**
Tecnica para evaluar un modelo dividiendo los datos en partes, entrenando con unas y evaluando con otras.

**CSV**
Comma-Separated Values. Archivo de texto con datos separados por comas. Formato comun para datasets.

---

## D

**Data Drift (Deriva de datos)**
Cambio en la distribucion de los datos con el tiempo. Puede hacer que un modelo que funcionaba bien deje de funcionar.

**Dataset (Conjunto de datos)**
Tabla de datos con filas (muestras) y columnas (caracteristicas + target).

**Deep Learning (Aprendizaje profundo)**
ML usando redes neuronales con muchas capas (no usado en este software).

**Desviacion Estandar (Standard Deviation)**
Mide que tan dispersos estan los valores respecto al promedio. Formula: `sqrt(Suma(xi - media)^2 / (n-1))`.

---

## E

**Ensemble (Conjunto)**
Tecnica que combina multiples modelos para obtener mejores predicciones que cada uno por separado. Ejemplos: Random Forest, Gradient Boosting.

**Entrenamiento (Training)**
Proceso donde el modelo "aprende" de los datos ajustando sus parametros internos.

**Entropia (Entropy)**
Measure de impureza en un arbol de decision. Entropia = 0 significa que todas las muestras son de la misma clase.

**Epoch (Epoca)**
Una pasada completa por todos los datos de entrenamiento (mas comun en redes neuronales).

**Error Absoluto Medio (MAE)**
Promedio de los errores absolutos. Formula: `(1/n) * Suma(|yi - yi_pred|)`.

**Error Cuadratico Medio (MSE)**
Promedio de los errores al cuadrado. Formula: `(1/n) * Suma((yi - yi_pred)^2)`.

---

## F

**Feature (Caracteristica/Variable)**
Columna de entrada que el modelo usa para hacer predicciones. Ejemplo: "metros cuadrados", "edad".

**Feature Importance (Importancia de caracteristica)**
Medida de que tan importante es cada variable para las predicciones del modelo.

**Fold (Pliegue)**
Una de las partes en que se dividen los datos en la validacion cruzada.

**Function Loss (Funcion de perdida)**
Funcion que mide que tan mal esta prediciendo el modelo. El entrenamiento busca minimizar esta funcion.

---

## G

**Gini Index (Indice de Gini)**
Medida de impureza en arboles de decision. Similar a la entropia. Formula: `1 - Suma(pk^2)`.

**GPU (Graphics Processing Unit)**
Unidad de procesamiento grafico. Algunos modelos pueden entrenarse mas rapido en GPU (XGBoost lo soporta).

---

## H

**Hiperparametro (Hyperparameter)**
Configuracion del modelo que se define **antes** del entrenamiento y no se aprende de los datos. Ejemplo: numero de arboles, profundidad maxima.

**HPO (Hyperparameter Optimization)**
Proceso de buscar los mejores hiperparametros automaticamente. En este software se usa Optuna.

---

## I

**Imputacion**
Tecnica para reemplazar valores faltantes con un valor estimado (media, mediana, moda, etc.).

**Inteligencia Artificial (AI)**
Campo de la computacion que busca crear sistemas que puedan realizar tareas que normalmente requieren inteligencia humana.

**Intervalo de Confianza (Confidence Interval)**
Rango de valores donde es probable que se encuentre el verdadero valor del parametro. Ejemplo: "IC 95% = [0.82, 0.86]" significa que estamos 95% seguros de que el valor real esta ahi.

---

## J

**JSON (JavaScript Object Notation)**
Formato de datos ligero, facil de leer y escribir. Usado para la comunicacion entre servicios del software.

---

## K

**K-Fold Cross-Validation**
Forma especifica de validacion cruzada donde los datos se dividen en K partes iguales.

**Kappa de Cohen (Cohen's Kappa)**
Metrica de clasificacion que mide el acuerdo entre predicciones y valores reales, corregido por el acuerdo esperado por azar. Va de -1 a +1.

---

## L

**Label (Etiqueta)**
Valor de la variable objetivo. Ejemplo: "aprobado", "reprobado" (clasificacion) o 150000 (regresion).

**Label Encoder**
Transformador que convierte etiquetas de texto a numeros (ej: "gato" -> 0, "perro" -> 1, "pez" -> 2).

**Lasso (L1)**
Variante de regresion lineal con penalizacion que puede eliminar variables (coeficientes = 0).

**Latex**
Sistema de composicion tipografica usado para documentos cientificos. El software genera informes en LaTeX.

**Learning Rate (Tasa de aprendizaje)**
Hiperparametro que controla que tanto ajusta el modelo sus pesos en cada paso. Valores bajos = aprendizaje lento pero preciso.

**Linear Regression (Regresion Lineal)**
Modelo que ajusta una linea recta a los datos.

---

## M

**Machine Learning (ML)**
Ver "Aprendizaje Automatico".

**MCC (Matthews Correlation Coefficient)**
Metrica de clasificacion que considera VP, VN, FP, FN. Considerada una de las mejores metricas para datos desbalanceados. Va de -1 a +1.

**Metrica**
Medida numerica que evalua que tan bueno es el modelo.

**Modelo**
Objeto matematico que ha aprendido a hacer predicciones a partir de datos.

**Moda (Mode)**
Valor que mas se repite en un conjunto de datos.

---

## N

**NaN (Not a Number)**
Representacion de un valor faltante o indefinido en programacion.

**Normalizacion**
Proceso de escalar los datos para que esten en un rango especifico (comunmente 0 a 1).

---

## O

**One-Hot Encoding**
Tecnica para convertir variables categoricas en columnas binarias (0 y 1). Ejemplo: "color_rojo" = 0 o 1, "color_azul" = 0 o 1.

**One-vs-Rest (OvR)**
Estrategia para clasificacion multiclase: entrena un modelo binario para cada clase contra todas las demas.

**Optuna**
Libreria de Python para optimizacion de hiperparametros. Usa el algoritmo TPE (Tree-structured Parzen Estimator).

**Outlier (Valor atipico)**
Dato que esta significativamente alejado del resto. Puede distorsionar los modelos.

**Overfitting (Sobreajuste)**
Cuando el modelo "memoriza" los datos de entrenamiento pero falla con datos nuevos. Se detecta cuando el modelo es perfecto en train pero malo en test.

---

## P

**Pipeline**
Cadena de transformaciones y modelo que se ejecutan en secuencia. Ejemplo: imputar -> escalar -> modelo.

**Precision**
De todas las veces que el modelo dijo "positivo", cuantas realmente lo erian. Formula: `VP / (VP + FP)`.

**Prediccion (Prediction)**
El resultado que el modelo genera para una entrada nueva.

**Preprocesamiento (Preprocessing)**
Transformacion de los datos antes de entrenar el modelo (imputacion, escalado, encoding, etc.).

---

## R

**R-cuadrado (R^2)**
Metrica de regresion que mide que porcentaje de la variacion del target es explicada por el modelo. Va de -infinito a 1.

**Random Forest**
Modelo de ensemble que construye muchos arboles de decision y promedia sus predicciones.

**Recall (Sensibilidad)**
De todos los que realmente son positivos, cuantos detecto el modelo. Formula: `VP / (VP + FN)`.

**Regresion**
Tipo de problema de ML donde se predice un valor numerico continuo.

**Regresion Logistica**
Modelo de clasificacion que usa la funcion sigmoid para predecir probabilidades.

**Regularizacion**
Tecnica para evitar sobreajuste penalizando modelos complejos. Ridge usa L2, Lasso usa L1.

**Residuo (Residual)**
Diferencia entre el valor real y el predicho. Formula: `residuo = yi - yi_pred`.

**Ridge (L2)**
Variante de regresion lineal con penalizacion que reduce los coeficientes pero no los elimina.

**RMSE (Root Mean Squared Error)**
Raiz del error cuadratico medio. Mide el error promedio en la misma unidad que el target. Formula: `sqrt((1/n) * Suma((yi - yi_pred)^2))`.

**ROC AUC (Area Under the ROC Curve)**
Metrica de clasificacion que mide la capacidad del modelo para distinguir entre clases. Va de 0 a 1.

---

## S

**Sigmoid (Sigmoide)**
Funcion matematica que convierte cualquier numero en un valor entre 0 y 1. Formula: `1 / (1 + e^(-z))`.

**StandardScaler**
Transformador que estandariza los datos: media = 0, desviacion = 1.

**Stratified (Estratificado)**
Division de datos que mantiene la proporcion de clases en cada fold. Importante para clasificacion.

**Supervisado (Supervised)**
Tipo de ML donde los datos incluyen la respuesta correcta (target).

---

## T

**Target (Objetivo)**
Variable que se quiere predecir. Tambien llamada variable dependiente o label.

**Test Set (Conjunto de prueba)**
Datos que el modelo **nunca vio** durante el entrenamiento, usados para evaluar su rendimiento real.

**TPE (Tree-structured Parzen Estimator)**
Algoritmo de busqueda inteligente usado por Optuna para encontrar los mejores hiperparametros.

**Train Set (Conjunto de entrenamiento)**
Datos usados para entrenar el modelo.

---

## U

**Underfitting (Subajuste)**
Cuando el modelo es demasiado simple y no puede capturar los patrones de los datos. Se detecta cuando es malo tanto en train como en test.

**Unsupervised (No supervisado)**
Tipo de ML donde los datos NO incluyen la respuesta correcta. Ejemplo: agrupar clientes por comportamiento (no usado en este software).

---

## V

**Validacion Cruzada (Cross-Validation)**
Ver "Cross-Validation".

**Variable Dependiente**
Ver "Target".

**Variable Independiente**
Ver "Feature".

**Variance (Varianza)**
Error del modelo por ser demasiado complejo y sensibles a los datos de entrenamiento. Opuesto al bias.

---

## W

**Weighted (Ponderado)**
Promedio donde cada valor tiene un peso diferente. Ejemplo: F1 weighted promedia el F1 de cada clase segun cuantas muestras tiene.

---

## X

**XGBoost (Extreme Gradient Boosting)**
Algoritmo de boosting optimizado, uno de los mas usados en competiciones y la industria. Version mejorada de Gradient Boosting.

---

## Z

**Z-score**
Valor estandarizado que indica cuantas desviaciones estandar esta un dato del promedio. Formula: `z = (x - media) / desviacion_estandar`.
