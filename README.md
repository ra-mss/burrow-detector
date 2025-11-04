# Detección de Madrigueras con YOLOv8 y AWS

Este proyecto es un pipeline de MLOps (Machine Learning Operations) de principio a fin para detectar y contar objetos diminutos (madrigueras) en imágenes TIF de alta resolución (+1.5 GB) usando Deep Learning.

El flujo de trabajo completo pasa de un etiquetado manual en QGIS a un pipeline automatizado de entrenamiento e inferencia desplegable en **AWS (Amazon SageMaker)**, usando GitHub para el control de versiones.

### Resultado final
![Resultado en QGIS](https://github.com/ra-mss/Ramses_Portfolio/blob/main/BD1.png)

---

## 🚀 Objetivo
El objetivo era contar miles de madrigueras de pocos píxeles en una imagen TIF de 1.58 GB (27497 x 29283 px). El análisis local en un notebook es inviable o muy lento (más de 1.5 horas solo para la inferencia en CPU).

## 💡 Solución
Desarrollé un pipeline modular en Python que divide el problema en fases, diseñado para ejecutarse en la nube:

1.  **Datos (S3):** Todos los archivos pesados (TIF, GPKG) se almacenan en **AWS S3**.
2.  **Fase 1 (Etiquetado en QGIS):** Se etiqueta un *recorte representativo* de la imagen (no la imagen completa).
3.  **Fase 2 (Crear Dataset):** `fase_2_crear_dataset.py` lee el TIF y los labels desde S3, genera miles de parches de `512x512` y los guarda de nuevo en S3.
4.  **Fase 3 (Entrenamiento):** Un **AWS SageMaker Training Job** toma los parches de S3, entrena un modelo `yolov8s` en una instancia con GPU (ej. `ml.g4dn.xlarge`), y guarda el modelo final (`best.pt`) en S3.
5.  **Fase 4 (Inferencia):** Un **AWS SageMaker Batch Transform Job** usa el modelo entrenado para escanear la imagen TIF *gigante* original desde S3, aplicando la inferencia en ventana deslizante (`sliding-window`) y fusionando los resultados con NMS.
6.  **Resultados (S3 y QGIS):** El script de inferencia guarda el conteo final (¡más de 21,000 detecciones!) y un archivo `.gpkg` en S3, listo para su visualización.

---

## 📊 Resultados del Modelo
El modelo final (`yolov8s` @ 100 épocas) alcanzó métricas excelentes:

* **mAP50:** `0.948 (94.8%)`
* **mAP50-95:** `0.578 (57.8%)`
* **Precisión:** `0.893 (89.3%)`
* **Recall:** `0.902 (90.2%)`

---

## ⚙️ Cómo Usar este Repositorio

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/proyecto-madrigueras.git](https://github.com/tu-usuario/proyecto-madrigueras.git)
    cd proyecto-madrigueras
    ```

2.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Descargar Datos:**
    Los datos (TIF, GPKG) y el modelo pre-entrenado (`best.pt`) están alojados en Google Drive:
    **[Descargar Datos y Modelo desde Google Drive](https://TU-ENLACE-DE-GOOGLE-DRIVE-AQUI)**

4.  **Configurar Rutas:**
    Modifica el archivo `config.py` para apuntar a las rutas donde descargaste los datos en tu máquina local o en tu bucket de S3.

5.  **Ejecutar el Pipeline:**
    ```bash
    # Fase 2: Crear el dataset desde cero
    python fase_2_crear_dataset.py

    # Fase 3: Entrenar el modelo desde cero
    python fase_3_entrenar.py

    # Fase 4: Ejecutar la inferencia con el modelo pre-entrenado
    python fase_4_inferencia.py
    ```