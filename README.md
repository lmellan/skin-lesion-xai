 
# Skin Lesion XAI ResNet50

Este repositorio contiene el notebook utilizado para ejecutar un modelo ResNet50 entrenado con imágenes del dataset PAD-UFES-20. El notebook permite realizar inferencia sobre imágenes de lesiones cutáneas y visualizar mapas Grad-CAM para observar qué zonas de la imagen influyen en la predicción del modelo.

## Importante sobre el notebook

El notebook fue desarrollado en Google Colab, por lo que se recomienda abrirlo y ejecutarlo directamente en ese entorno.

Es posible que GitHub no lo muestre correctamente y aparezca el mensaje `Invalid Notebook`. Esto puede ocurrir por metadatos internos generados por Colab o por salidas interactivas del notebook. En ese caso, el archivo no necesariamente está dañado.

Para revisarlo correctamente:

1. Descargar el archivo `.ipynb` desde el repositorio.
2. Abrirlo en Google Colab.
3. Ejecutar las celdas desde Colab.

## Archivo principal

```text
pipeline_resnet50_padufes_gradcam.ipynb
````

## Contenido general

El notebook incluye la descarga y preparación del dataset, entrenamiento/configuración del modelo ResNet50, inferencia sobre imágenes de lesiones cutáneas, evaluación de resultados y generación de mapas Grad-CAM.

## Archivos y carpetas generadas

Durante la ejecución del notebook se crean distintas carpetas y archivos:

```text
data/
```

Contiene los datos utilizados por el notebook.

* `data/raw/`: almacena los archivos descargados originalmente, como el archivo de metadatos y los `.zip` con imágenes.
* `data/images/`: contiene las imágenes extraídas del dataset.
* `data/metadata.csv`: archivo con la información asociada a las imágenes.

```text
models/
```

Contiene el modelo entrenado o configurado.

* `models/resnet_configurado.pt`: checkpoint del modelo ResNet50 utilizado para realizar inferencia.

```text
outputs/
```

Contiene los resultados generados por el notebook.

* `outputs/splits.json`: división de datos utilizada para entrenamiento, validación y prueba.
* `outputs/samples_grid.png`: visualización de ejemplos del dataset.
* `outputs/training_curves.png`: curvas de entrenamiento.
* `outputs/training_history.csv`: historial numérico del entrenamiento.
* `outputs/metrics.json`: métricas obtenidas por el modelo.
* `outputs/confusion_matrix.png`: matriz de confusión binaria.
* `outputs/confusion_matrix_6way.png`: matriz de confusión multiclase.
* `outputs/roc_curve.png`: curva ROC.
* `outputs/binary_confusion_and_roc.png`: resumen visual con matriz de confusión y curva ROC.

```text
outputs/gradcam/
```

Contiene las imágenes generadas con Grad-CAM.

* `gradcam_*.png`: visualizaciones donde se muestra la zona de la imagen que más influyó en la predicción del modelo.

 
 
