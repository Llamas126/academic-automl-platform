# Conceptos Basicos de Machine Learning

## Que es Machine Learning?

**Machine Learning** (Aprendizaje Automatico) es una rama de la inteligencia artificial donde las computadoras **aprenden patrones a partir de datos**, en lugar de seguir instrucciones explicitas.

### Analogia simple

Pensemos en aprender a reconocer frutas:

- **Programacion tradicional:** Escribimos reglas como "si es redondo y verde, es una manzana". Esto falla porque hay manzanas rojas y peras verdes.
- **Machine Learning:** Le mostramos a la computadora 1000 fotos de manzanas y 1000 fotos de peras. La computadora **aprende sola** que patron visual distingue una de otra.

## Aprendizaje Supervisado

El tipo de machine learning que usa este software es **supervisado**, lo que significa que:

- Tenemos **datos con respuesta conocida** (el "resultado correcto")
- La computadora aprende a predecir ese resultado a partir de otros datos

### Ejemplo

| Metro^2 | Habitaciones | Zona | **Precio** (lo que queremos predecir) |
|---------|-------------|------|--------------------------------------|
| 80 | 3 | Norte | $150,000 |
| 120 | 4 | Sur | $220,000 |
| 60 | 2 | Este | $110,000 |

Los tres primeros campos son las **features** (caracteristicas) y el precio es el **target** (objetivo/variable a predecir).

## Dos tipos de problema

### 1. Regresion: Predecir un numero

Cuando queremos predecir un valor continuo (un numero), es **regresion**.

- Predecir el precio de una casa
- Predecir la temperatura de manana
- Predecir la nota de un examen

Matematicamente, buscamos una funcion:

```
y = f(x1, x2, x3, ...)
```

Donde `y` es el valor a predecir y `x1, x2, x3...` son las caracteristicas.

**En este software:** Si el target tiene mas de 20 valores unicos o mas del 5% de filas tienen valores unicos, se asume regresion.

### 2. Clasificacion: Predecir una categoria

Cuando queremos predecir a que grupo pertenece algo, es **clasificacion**.

- Predecir si un email es spam o no spam
- Predecir si un tumor es benigno o maligno
- Predecir si un estudiante aprueba o reprueba

Matematicamente, buscamos:

```
y = clase(x1, x2, x3, ...)
```

Donde `y` es una categoria (0 o 1, o 0/1/2/3...).

**En este software:** Si el target es categorico o numerico con pocos valores unicos (2 a 50 clases), se asume clasificacion.

## Como el software decide que tipo de problema es

El software analiza la columna objetivo automaticamente:

```
Es numerico?
├── Si, y tiene >20 valores unicos o >5% unicos
│   └── REGRESION
├── Si, y tiene entre 2 y 10 valores unicos
│   └── CLASIFICACION
└── No es numerico (texto)
    ├── Tiene entre 2 y 50 clases
    │   └── CLASIFICACION
    └── Tiene mas de 50 clases
        └── Error (demasiadas categorias)
```

## Terminos clave

| Termino | Que significa | Ejemplo |
|---------|---------------|---------|
| **Dataset** | Conjunto de datos (tabla) | Una hoja de Excel con 1000 filas |
| **Feature** | Columna de entrada (variable independiente) | "metros cuadrados", "edad" |
| **Target** | Columna a predecir (variable dependiente) | "precio", "aprobado/reprobado" |
| **Muestra/Fila** | Cada registro individual | Una casa, un paciente, un estudiante |
| **Entrenamiento** | Proceso de aprender de los datos | El modelo "estudia" los datos |
| **Prediccion** | Lo que el modelo dice que sera | "Esta casa costara $180,000" |
| **Error** | Diferencia entre lo predicho y lo real | Predije $180k, cuesta $200k = error de $20k |
