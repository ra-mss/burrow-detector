# config.py
"""
Archivo de Configuración Centralizado para el pipeline de detección.
Modifica estas rutas para apuntar a tus datos locales o a buckets de S3 en AWS.
"""

# --- PARÁMETROS DE PROCESAMIENTO ---
PATCH_SIZE = 512
STRIDE = 256
PROB_GUARDAR_VACIO = 0.1
VAL_SPLIT = 0.2

# --- PARÁMETROS DE ENTRENAMIENTO ---
MODELO_YOLO_BASE = 'yolov8s.pt'  # (n, s, m, l, x)
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 512
NOMBRE_PROYECTO_YOLO = 'madrigueras_yolov8s'

# --- PARÁMETROS DE INFERENCIA ---
CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.45

# --- RUTAS DE ARCHIVOS ---

# Rutas de Google Drive (Configuración actual en Colab)
# (Cuando uses AWS, cambiar esto a "s3://tu-bucket/...")
DRIVE_BASE_DIR = "/content/drive/MyDrive/CNN"

# Fase 1 y 2: Archivos de entrada para crear el dataset
RUTA_TIF_RECORTE_ENTRENAR = f"{DRIVE_BASE_DIR}/seccion2_recorte_labels.tif"
RUTA_LABELS_GPKG = f"{DRIVE_BASE_DIR}/labels.gpkg"
CAMPO_CLASE_EN_GPKG = "label"  # El nombre de la columna en tu GPKG
CLASE_OBJETIVO = "1"         # El valor de la etiqueta en esa columna

# Fase 2: Salida
RUTA_DATASET_PARCHES = f"{DRIVE_BASE_DIR}/dataset_madrigueras"

# Fase 3: Entrada y salida
RUTA_DATASET_SPLIT = f"{DRIVE_BASE_DIR}/dataset_split"
RUTA_DATA_YAML = f"{DRIVE_BASE_DIR}/data.yaml"
# (La ruta final del modelo se genera automáticamente por YOLO en /runs/ 
#  El script de inferencia la lee desde aquí)
RUTA_MODELO_ENTRENADO = f"/content/runs/detect/{NOMBRE_PROYECTO_YOLO}/weights/best.pt"
RUTA_MODELO_GUARDADO = f"{DRIVE_BASE_DIR}/burrow_best_yolov8s.pt" # Tu copia en Drive

# Fase 4: Entrada y Salida
RUTA_TIF_GIGANTE_INFERENCIA = f"{DRIVE_BASE_DIR}/seccion_2.tif"
RUTA_RESULTADOS_GPKG = f"{DRIVE_BASE_DIR}/resultados_deteccion_seccion_2.gpkg"