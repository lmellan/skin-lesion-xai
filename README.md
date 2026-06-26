# Explicabilidad ResNet50 con Grad-CAM en lesiones cutaneas

Repositorio final del proyecto de Inteligencia Artificial Explicable (XAI) para analizar un modelo ResNet50 aplicado a imagenes de lesiones cutaneas del dataset PAD-UFES-20.

El objetivo principal del proyecto es estudiar si las predicciones del modelo pueden interpretarse mediante Grad-CAM y evaluarse con analisis complementarios de correspondencia espacial y fidelidad.

## Archivos del repositorio

```text
README.md
pipeline_resnet50_padufes_gradcam.ipynb
pipeline_resnet50_padufes_gradcam.pdf
```

- `pipeline_resnet50_padufes_gradcam.ipynb`: notebook final ejecutado. Contiene el pipeline completo del proyecto, incluyendo preparacion de datos, entrenamiento/evaluacion del modelo, generacion de Grad-CAM, analisis con cajas manuales y calculo de fidelidad.
- `pipeline_resnet50_padufes_gradcam.pdf`: exportacion del notebook final en PDF, util para revisar rapidamente la ejecucion y los resultados sin abrir Colab/Jupyter.
- `README.md`: descripcion general del repositorio y guia de uso.

## Notebook principal

La version final ejecutada del proyecto es:

```text
pipeline_resnet50_padufes_gradcam.ipynb
```

Este archivo es la referencia principal del repositorio. Si se quiere reproducir el trabajo, debe abrirse y ejecutarse en Google Colab.

## Contenido del notebook

El notebook incluye:

- Carga y preparacion del dataset PAD-UFES-20.
- Division de datos por paciente para evitar fuga de informacion entre entrenamiento, validacion y prueba.
- Entrenamiento de una ResNet50 preentrenada en ImageNet.
- Evaluacion multiclase y evaluacion binaria cancer vs no cancer.
- Seleccion de umbral binario usando el conjunto de validacion.
- Generacion de mapas Grad-CAM.
- Analisis cualitativo de casos verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
- Analisis espacial con cajas manuales sobre la lesion.
- Calculo de Faithfulness Correlation como metrica cuantitativa de fidelidad.
- Registro del entorno de ejecucion usado en Google Colab.

## Resultados principales

Evaluacion binaria en test: cancer (`BCC`, `MEL`, `SCC`) vs no cancer (`ACK`, `NEV`, `SEK`).

- Imagenes de test: `460`.
- Accuracy: `0.7913`.
- Balanced accuracy: `0.7932`.
- Precision cancer: `0.7490`.
- Sensibilidad cancer: `0.8565`.
- Especificidad: `0.7300`.
- F1 cancer: `0.7992`.
- ROC-AUC: `0.8765`.
- PR-AUC: `0.8597`.

Resultados de explicabilidad:

- Casos Grad-CAM analizados cualitativamente: `20`.
- Casos con cajas manuales validas: `20`.
- Activacion media dentro de la lesion: `0.5201`.
- Enriquecimiento medio de activacion dentro de la lesion: `2.8766`.
- Faithfulness Correlation media: `0.0640`.
- Faithfulness Correlation mediana: `0.0514`.
- Casos validos para fidelidad: `32`.

En conjunto, los resultados muestran que Grad-CAM entrega una interpretacion visual util, pero parcial. La activacion suele concentrarse mas en la lesion en casos correctamente clasificados, aunque tambien aparecen mapas difusos o casos donde mirar la region de la lesion no garantiza una decision correcta. La baja Faithfulness Correlation indica que una explicacion visualmente plausible no necesariamente refleja de forma fiel el comportamiento causal del modelo.

## Entorno de ejecucion

El notebook fue ejecutado en Google Colab con:

- Python `3.12.13`.
- GPU Tesla T4 con `15.64 GB` de VRAM.
- 2 CPU.
- CUDA `12.8`.
- PyTorch `2.11.0+cu128`.
- torchvision `0.26.0+cu128`.
- Quantus `0.6.0`.
- scikit-learn `1.6.1`.
- NumPy `2.0.2`.
- pandas `2.2.2`.
- matplotlib `3.10.0`.
- Pillow `11.3.0`.

## Uso recomendado

1. Abrir `pipeline_resnet50_padufes_gradcam.ipynb` en Google Colab.
2. Ejecutar las celdas en orden si se desea reproducir el pipeline.
3. Revisar `pipeline_resnet50_padufes_gradcam.pdf` si solo se quiere inspeccionar la ejecucion final y sus resultados.

 
