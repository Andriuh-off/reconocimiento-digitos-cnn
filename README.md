# Reconocimiento de dígitos manuscritos para formularios (caso TecnoForms)

Trabajo final de Machine Learning y Deep Learning con Python.

**Autor:** [Apellidos y nombres]
**Curso:** [Nombre del curso]
**Institución:** SENATI

## Caso práctico

TecnoForms necesita automatizar la lectura de formularios físicos escritos a mano (inscripción, pagos, solicitudes), sobre todo en educación y finanzas. Hoy los dígitos se leen y digitan de forma manual, lo que es lento y propenso a errores.

Este proyecto desarrolla y evalúa la etapa de **reconocimiento de dígitos manuscritos (0 al 9)** con redes neuronales en TensorFlow/Keras.

## Qué contiene el proyecto

| Parte del notebook | Contenido |
|---|---|
| Parte 1 | Carga de MNIST y preparación de imágenes de un formulario simulado (escala de grises, binarización, segmentación y normalización) |
| Parte 2 | Partición de datos, normalización y dos redes neuronales densas |
| Parte 3 | Red neuronal convolucional (CNN), validación cruzada y ajuste de neuronas |
| Parte 4 | Evaluación en el conjunto de prueba y comparación de los 3 modelos |
| Parte 5 | Guardado de los modelos para la app del navegador |

## Datos

- **Dataset:** MNIST (70 000 imágenes de dígitos de 28×28 píxeles), descargado con `fetch_openml`.
- **Partición:** 80 % entrenamiento (56 000) y 20 % prueba (14 000), con `random_state=42`.
- **Normalización:** los píxeles se dividen entre 255 para quedar entre 0 y 1.

## Preparación de imágenes del formulario

1. Conversión a escala de grises y suavizado.
2. Binarización con el método de Otsu (dígito blanco sobre fondo negro).
3. Segmentación de cada dígito por contornos.
4. Redimensionado a 20×20 y centrado en un lienzo de 28×28.
5. Normalización entre 0 y 1.

Se probó con un formulario simulado de cinco dígitos, que se separaron correctamente.

## Modelos

| Modelo | Arquitectura | Entrenamiento |
|---|---|---|
| Red densa 1 | Entrada de 784 valores, Dense 128 (ReLU), Dense 64 (ReLU), Dense 10 (Softmax) | 10 épocas, Adam |
| Red densa 2 | Entrada de 784 valores, Dense 64 (ReLU), Dense 32 (ReLU), Dense 10 (Softmax) | Adam, `EarlyStopping` |
| CNN (principal) | 3 bloques Conv2D (16, 32 y 64 filtros, kernel 3×3, ReLU) con MaxPooling, Flatten, Dense 64 (ReLU), Dense 10 (Softmax). 28 106 parámetros | Adam, `EarlyStopping` |

Función de pérdida en todos: `sparse_categorical_crossentropy`.

## Validación y ajuste

- **Validación cruzada:** `KFold` de 3 partes, sobre un subconjunto de 6 000 imágenes, con una versión reducida de la CNN (2 capas convolucionales).
- **Ajuste de hiperparámetros:** se probaron 32, 64 y 128 neuronas en la capa densa.
- El conjunto de prueba se usó una sola vez, al final.

## Resultados en el conjunto de prueba

| Modelo | Exactitud |
|---|---|
| Red densa 1 | 97.49 % |
| Red densa 2 | 96.78 % |
| CNN | **98.16 %** |

La CNN cometió **257 errores en 14 000 imágenes**. Su reporte de métricas por dígito (precisión, recall y F1), la matriz de confusión y los gráficos de predicciones frente a etiquetas reales están en las Partes 3 y 4 del notebook.

## Conclusiones y limitaciones

- La CNN fue el mejor modelo, aunque con una ventaja moderada sobre las redes densas, porque conserva la estructura bidimensional de la imagen.
- MNIST tiene dígitos más limpios y centrados que los formularios reales, por lo que la precisión podría bajar con letra más variada, manchas o ruido.
- En datos críticos (montos, documentos) se recomienda una revisión humana.
- Mejoras futuras: reentrenar con dígitos de formularios reales, aumento de datos y probar otras configuraciones de la red.

## Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install numpy matplotlib opencv-python scikit-learn tensorflow flask
```

### 2. Ejecutar el notebook

Abre `entregable.ipynb` y ejecuta todas las celdas en orden. La Parte 5 crea la carpeta `modelos/` con los tres modelos entrenados.

### 3. App del navegador (opcional)

Dibuja un dígito con el mouse y los 3 modelos lo leen. Todo corre en tu computadora, sin internet.

```bash
cd app_digitos
python app.py
```

Abre `http://127.0.0.1:5000`. La carpeta `modelos/` debe estar junto a `app.py`.

## Estructura del repositorio

```
├── README.md
├── entregable.ipynb
└── app_digitos/
    ├── app.py
    ├── static/
    │   └── index.html
    └── modelos/          (se crea con la Parte 5 del notebook)
```

## Tecnologías

Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, OpenCV y Flask.
