import os
import glob
import random
import shutil
import subprocess
import config  # Importamos config

def dividir_dataset():
    """Divide los parches de Fase 2 en train/val."""
    print("--- Iniciando División de Dataset ---")
    DATASET_IN_DIR = config.RUTA_DATASET_PARCHES
    DATASET_OUT_DIR = config.RUTA_DATASET_SPLIT

    dirs_to_create = [
        os.path.join(DATASET_OUT_DIR, "images", "train"),
        os.path.join(DATASET_OUT_DIR, "images", "val"),
        os.path.join(DATASET_OUT_DIR, "labels", "train"),
        os.path.join(DATASET_OUT_DIR, "labels", "val"),
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    print(f"Directorios creados en: {DATASET_OUT_DIR}")

    all_images = glob.glob(os.path.join(DATASET_IN_DIR, "images", "*.png"))
    random.shuffle(all_images)

    split_point = int(len(all_images) * (1 - config.VAL_SPLIT))
    train_images = all_images[:split_point]
    val_images = all_images[split_point:]

    print(f"Dataset total: {len(all_images)} imágenes.")
    print(f"  -> Entrenamiento: {len(train_images)} imágenes.")
    print(f"  -> Validación: {len(val_images)} imágenes.")

    def copy_files(file_list, set_name):
        for img_path in file_list:
            try:
                basename = os.path.basename(img_path)
                label_name = os.path.splitext(basename)[0] + ".txt"
                label_path = os.path.join(DATASET_IN_DIR, "labels", label_name)
                
                dest_img_path = os.path.join(DATASET_OUT_DIR, "images", set_name, basename)
                dest_label_path = os.path.join(DATASET_OUT_DIR, "labels", set_name, label_name)
                
                shutil.copy(img_path, dest_img_path)
                if os.path.exists(label_path):
                    shutil.copy(label_path, dest_label_path)
                else:
                    open(dest_label_path, 'w').close()
            except Exception as e:
                print(f"Error copiando {img_path}: {e}")

    print("Copiando archivos de entrenamiento...")
    copy_files(train_images, "train")
    print("Copiando archivos de validación...")
    copy_files(val_images, "val")
    print("¡División del dataset completada!")

def crear_yaml():
    """Crea el archivo data.yaml para YOLO."""
    print(f"--- Creando archivo data.yaml en {config.RUTA_DATA_YAML} ---")
    
    contenido_yaml = f"""
path: {config.RUTA_DATASET_SPLIT}
train: images/train
val: images/val

# Clases
names:
  0: burrow
"""
    with open(config.RUTA_DATA_YAML, 'w') as f:
        f.write(contenido_yaml)
    print("¡Archivo data.yaml creado!")

def entrenar_modelo():
    """Ejecuta el comando de entrenamiento de YOLO."""
    print("--- Iniciando Fase 3: Entrenamiento del Modelo ---")
    
    comando = [
        "yolo", "task=detect", "mode=train",
        f"model={config.MODELO_YOLO_BASE}",
        f"data={config.RUTA_DATA_YAML}",
        f"epochs={config.EPOCHS}",
        f"imgsz={config.IMG_SIZE}",
        f"batch={config.BATCH_SIZE}",
        f"name={config.NOMBRE_PROYECTO_YOLO}"
    ]
    
    print(f"Ejecutando comando: {' '.join(comando)}")
    try:
        subprocess.run(comando, check=True)
        print("--- ¡Entrenamiento Completado! ---")
        
        # Copiar el modelo final a la carpeta de Drive
        ruta_original_modelo = config.RUTA_MODELO_ENTRENADO
        ruta_destino_modelo = config.RUTA_MODELO_GUARDADO
        print(f"Copiando modelo final a: {ruta_destino_modelo}")
        shutil.copy(ruta_original_modelo, ruta_destino_modelo)
        print("¡Modelo copiado exitosamente!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error durante el entrenamiento: {e}")
    except FileNotFoundError:
        print(f"Error: No se encontró el modelo en {ruta_original_modelo}")
        print("Verifica el nombre del proyecto en config.py")

if __name__ == "__main__":
    dividir_dataset()
    crear_yaml()
    entrenar_modelo()